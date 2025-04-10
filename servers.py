import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np
import os
from torch._prims_common import TensorSequenceType
from ultralytics import YOLO
import torch
import yaml

class YOLOStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        print(f"\nRound {rnd}: Aggregating results from {len(results)} clients (failures: {len(failures)})")
        
        # Call aggregate_fit from base class (FedAvg)
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Round {rnd}: Aggregation successful")
            
            # Calculate mean of metrics
            accuracies = [r.metrics["map50"] for _, r in results]
            losses = [r.metrics["loss"] for _, r in results]
            
            print(f"Round {rnd} metrics:")
            print(f"  - Mean mAP@50: {np.mean(accuracies):.2%}")
            print(f"  - Mean Loss: {np.mean(losses):.4f}")

            # Save model after final round
            if rnd == 2:  # Assuming 20 rounds as per ServerConfig
                save_dir = "saved_models"
                os.makedirs(save_dir, exist_ok=True)
                
                # Initialize a YOLO model
                model = YOLO('yolov8n.pt')
                
                 # Update the model with the correct number of classes
                model.model.model[-1].nc = 5  # Set number of classes to 5
                model.model.model[-1].no = model.model.model[-1].nc + 5  # Update number of outputs
                
                # Convert parameters to state dict
                state_dict = {}
                params_dict = model.model.state_dict()
                
                '''# Get the tensors from the aggregated parameters
                if isinstance(aggregated_parameters, tuple):
                    parameters_tensors = aggregated_parameters[0].tensors  # Get tensors from the Parameters object
                else:
                    parameters_tensors = aggregated_parameters.tensors
                
                # Convert tensors to NumPy arrays    
                parameters_list = [fl.common.bytes_to_ndarray(tensor) for tensor in parameters_tensors]
                
                # Load aggregated parameters
                for i, (name, _) in enumerate(params_dict.items()):
                    param_tensor = torch.from_numpy(parameters_li
                    st[i])
                    state_dict[name] = param_tensor'''
                
                # Get the tensors from the aggregated parameters
                if isinstance(aggregated_parameters, tuple):
                    parameters_tensors = aggregated_parameters[0].tensors
                else:
                    parameters_tensors = aggregated_parameters.tensors
                    
                parameters_list = [fl.common.bytes_to_ndarray(tensor) for tensor in parameters_tensors]
                
                # Create new state dict with matching shapes
                for i, ((name, param), aggregated_param) in enumerate(zip(params_dict.items(), parameters_list)):
                    if param.shape == aggregated_param.shape:
                        state_dict[name] = torch.from_numpy(aggregated_param)
                
                # Load aggregated parameters
                model.model.load_state_dict(state_dict, strict=False)
                
                # Save the model
                save_path = os.path.join(save_dir, "final_federated_model.pt")
                torch.save(model.model.state_dict(), save_path)
                print(f"\nFinal model saved to {save_path}")
                
        return aggregated_parameters
      
        
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict]]:
        print(f"\nRound {rnd}: Aggregating evaluation results")
    
        if not results:
            return None

    # Calculate mean of metrics
        val_metrics = [(r.loss, r.metrics["val_map50"]) for _, r in results]
        loss = np.mean([m[0] for m in val_metrics])
        map50 = np.mean([m[1] for m in val_metrics])
    
        print(f"Round {rnd} evaluation metrics:")
        print(f"  - Mean Validation Loss: {loss:.4f}")
        print(f"  - Mean Validation mAP@50: {map50:.2%}")
    
    # Return loss and metrics as a tuple
        return loss, {"val_map50": map50}

if __name__ == "__main__":
    # Create strategy instance
    strategy = YOLOStrategy(
        min_fit_clients=2,          # Minimum number of clients for training
        min_available_clients=2,     # Minimum number of available clients
        min_evaluate_clients=2,      # Minimum number of clients for evaluation
        fraction_fit=0.8,           # Fraction of clients used for training
        fraction_evaluate=0.5       # Fraction of clients used for evaluation
    )
    
    print("\nStarting Flower server...")
    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2),  # Increased number of rounds
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024  # 1GB
    )












'''def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        print(f"\nRound {rnd}: Aggregating evaluation results")
        
        if not results:
            return None

        # Calculate mean of metrics
        val_metrics = [(r.loss, r.metrics["val_map50"]) for _, r in results]
        loss = np.mean([m[0] for m in val_metrics])
        map50 = np.mean([m[1] for m in val_metrics])
        
        print(f"Round {rnd} evaluation metrics:")
        print(f"  - Mean Validation Loss: {loss:.4f}")
        print(f"  - Mean Validation mAP@50: {map50:.2%}")
        
        return loss'''