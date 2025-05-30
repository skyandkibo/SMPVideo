import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import csv
from sklearn.metrics import mean_absolute_percentage_error

# target_name_list=['popularity']

def custom_log(x):
    return np.log(1+abs(x))

def custom_exp(x):
    return np.exp(x) - 1
    
# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# # Call set_seed at the start
# set_seed(42)
# root_path = '/sss/Video'
# # Load training and testing data
# train_excel_file = f'{root_path}/train.xlsx'
# test_excel_file = f'{root_path}/test.xlsx'

# df_train = pd.read_excel(train_excel_file)
# df_test = pd.read_excel(test_excel_file)

# # Filter out invalid video_ids from df_train
# invalid_video_ids = ['VIDEO00001914']

# # Clean video_id and apply transformation
# df_train['vid'] = df_train['vid'].astype(str).str.replace("'", "")
# df_train = df_train[~df_train['vid'].isin(invalid_video_ids)]

# df_test['vid'] = df_test['vid'].astype(str).str.replace("'", "")

# # Define 5 different feature folder paths for training and testing
# train_folder_paths = [
#     f'{root_path}/repair_train_test/videomae_features_training', f'{root_path}/repair_train_test/vivit_features-training',
#     f'{root_path}/repair_train_test/timesformer-features-training', f'{root_path}/repair_train_test/xclip-large-patch14-kinetics-600-training',
#     f'{root_path}/repair_train_test/opengv_training_0/embeddings',
#     f'{root_path}/repair_train_test/lava_training_0/embeddings'
# ]

# test_folder_paths = [
#     f'{root_path}/repair_train_test/videomae_features_testing', f'{root_path}/repair_train_test/vivit_features-testing',
#     f'{root_path}/repair_train_test/timesformer-features-testing', f'{root_path}/repair_train_test/xclip-large-patch14-kinetics-600-testing',
#     f'{root_path}/repair_train_test/opengv_testing_0/embeddings',
#     f'{root_path}/repair_train_test/lava_testing_0/embeddings'
# ]

# Custom dataset to return 5 features and 1 target
class VideoDataset(Dataset):
    def __init__(self, dataframe, folder_paths, device='cpu',target_name='popularity'):
        self.dataframe = dataframe
        self.folder_paths = folder_paths  # Paths for 5 different folders
        self.device = device  # Store device info
        self.targets = self.dataframe[target_name].values.reshape(-1, 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        video_id = self.dataframe.iloc[idx]['vid']
        
        try:
            # Load features from 5 separate folders
            feature_tensors = []
            for folder in self.folder_paths:
                file_path = os.path.join(folder, f'{video_id}.pt')
                #print(file_path)
                if not os.path.exists(file_path):
                    if 'lava' in folder:
                        feature_tensor = torch.zeros(1, 1024).to(self.device)
                else:
                    feature_tensor = torch.load(file_path, weights_only=True)
                    feature_tensor = feature_tensor.to(self.device)
                
                feature_tensors.append(feature_tensor)

            # Unpack the feature tensors into separate variables
            feature1, feature2, feature3, feature4, feature5, feature6 = feature_tensors

            # Get the target value
            target = self.targets[idx]
            target_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)
            # print(target_tensor)

            return feature1, feature2, feature3, feature4, feature5, feature6, target_tensor

        except FileNotFoundError as e:
            print(f"Error: Could not find pt files for video_id {video_id}. Skipping this sample.")
            return None

# Neural network model

class VideoDataset_pred(Dataset):
    def __init__(self, dataframe, folder_paths, device='cpu',target_name='popularity'):
        self.dataframe = dataframe
        self.folder_paths = folder_paths
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        video_id = self.dataframe.iloc[idx]['vid']
        post_id = self.dataframe.iloc[idx]['pid']

        try:
            # Load features from 5 separate folders
            feature_tensors = []
            for folder in self.folder_paths:
                file_path = os.path.join(folder, f'{video_id}.pt')

                if not os.path.exists(file_path):
                    if 'lava' in folder:
                        feature_tensor = torch.zeros(1, 1024).to(self.device)
                else:
                    feature_tensor = torch.load(file_path, weights_only=True)
                    feature_tensor = feature_tensor.to(self.device)
                
                feature_tensors.append(feature_tensor)

            feature1, feature2, feature3, feature4, feature5, feature6 = feature_tensors
            

            return feature1, feature2, feature3, feature4, feature5, feature6, post_id

        except FileNotFoundError as e:
            return None

def pad_or_truncate(self, feature, max_len):
    """Fill or truncate the input feature sequence"""
    if feature.shape[0] < max_len:
        padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
        feature = torch.cat((feature, padding), dim=0)
    else:
        feature = feature[:max_len]
    return feature

# Custom collate function to handle 5 features and target
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return (torch.empty(0),) * 6

    feature1_list, feature2_list, feature3_list, feature4_list, feature5_list, feature6_list, targets_list = zip(*batch)

    feature1 = torch.stack(feature1_list)
    feature2 = torch.stack(feature2_list)
    feature3 = torch.stack(feature3_list)

    new_feature4_list = []
    for x in feature4_list:
        new_feature4_list.append(torch.mean(x, axis=0).unsqueeze(0))
    feature4 = torch.stack(new_feature4_list)

    feature5 = torch.stack(feature5_list)
    feature6 = torch.stack(feature6_list)
    targets = torch.stack(targets_list)

    return feature1, feature2, feature3, feature4, feature5, feature6, targets

def collate_fn_pred(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return (torch.empty(0),) * 7

    feature1_list, feature2_list, feature3_list, feature4_list, feature5_list, feature6_list, post_id_list = zip(*batch)

    feature1 = torch.stack(feature1_list)
    feature2 = torch.stack(feature2_list)
    feature3 = torch.stack(feature3_list)
    new_feature4_list = []
    for x in feature4_list:
        new_feature4_list.append(torch.mean(x, axis=0).unsqueeze(0))
    feature4 = torch.stack(new_feature4_list)
    feature5 = torch.stack(feature5_list)
    feature6 = torch.stack(feature6_list)
    post_ids = list(post_id_list)

    return feature1, feature2, feature3, feature4, feature5, feature6, post_ids
# Prepare dataloaders to return 5 features and 1 target
def prepare_dataloaders(df, folder_paths, batch_size=32, device='cpu',target_name='popularity'):
    dataset = VideoDataset(df, folder_paths, device=device,target_name=target_name)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

def prepare_dataloaders_pred(df, folder_paths, batch_size=32, device='cpu',target_name='popularity'):
    dataset = VideoDataset_pred(df, folder_paths, device=device,target_name=target_name)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pred)
    return data_loader

# Training loop
def train_model(model, train_loader, val_loader, save_path, logger, video_log, num_epochs=20, learning_rate=0.0001, device='cpu',target_name='popularity'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, num_train_batches = 0, 0
        train_predictions, train_targets_list = [], []
        for feature1, feature2, feature3, feature4, feature5, feature6, targets in train_loader:
            feature1, feature2, feature3, feature4, feature5, feature6 = (
                feature1.to(device), feature2.to(device), feature3.to(device),
                feature4.to(device), feature5.to(device), feature6.to(device)
            )
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(feature1, feature2, feature3, feature4, feature5, feature6)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1
                
            train_predictions.extend(outputs.squeeze().cpu().tolist())
            train_targets_list.extend(targets.squeeze().cpu().tolist())

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        
        if video_log:
            mape = mean_absolute_percentage_error(custom_exp(train_targets_list), custom_exp(train_predictions))
        else:
            mape = mean_absolute_percentage_error(train_targets_list, train_predictions)

        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Train loss: {avg_train_loss:.4f}, Train mape: {mape}')

        if val_loader is not None:
            model.eval()
            val_loss, num_val_batches = 0, 0
            predictions, targets_list = [], []
            with torch.no_grad():
                for feature1, feature2, feature3, feature4, feature5, feature6, targets in val_loader:
                    feature1, feature2, feature3, feature4, feature5, feature6 = (
                        feature1.to(device), feature2.to(device), feature3.to(device),
                        feature4.to(device), feature5.to(device), feature6.to(device)
                    )
                    targets = targets.to(device)
                    outputs = model(feature1, feature2, feature3, feature4, feature5, feature6)
                    val_loss += criterion(outputs, targets).item()
                    num_val_batches += 1
                    predictions.extend(outputs.squeeze().cpu().tolist())
                    targets_list.extend(targets.squeeze().cpu().tolist())
            
            if video_log:
                mape = mean_absolute_percentage_error(custom_exp(targets_list), custom_exp(predictions))
            else:
                mape = mean_absolute_percentage_error(targets_list, predictions)

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            logger.info(f'Epoch: {epoch+1}/{num_epochs}, Test loss: {avg_val_loss:.4f}, Test MAPE: {mape}')

    torch.save(model.state_dict(), save_path)
    logger.info(f'Model saved at {save_path}')

# Function to load the saved model
def load_saved_model(feature_sizes, model_path, logger, device='cpu',target_name='popularity'):
    model = VideoHeartPredictor(feature_sizes)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model state dictionary
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    logger.info(f"Model loaded from {model_path} for prediction.")
    return model

# Prediction function with post_id
def predict_and_save(model, test_loader, output_csv, logger, video_log, device='cpu',target_name='popularity'):
    model.eval()
    predictions = []
    post_ids = []
    with torch.no_grad():
        for feature1, feature2, feature3, feature4, feature5, feature6, post_id in test_loader:
            feature1, feature2, feature3, feature4, feature5, feature6 = (
                feature1.to(device), feature2.to(device), feature3.to(device),
                feature4.to(device), feature5.to(device), feature6.to(device)
            )
            # Forward pass
            outputs = model(feature1, feature2, feature3, feature4, feature5, feature6)

            # Inverse log transformation and rounding
            outputs = outputs.cpu().numpy().reshape(-1, 1)
            if video_log:
                outputs = custom_exp(outputs)  # Inverse of log transformation

            # Store the rounded predictions and corresponding video IDs
            predictions.extend(outputs)
            post_ids.extend(post_id)

    # Save predictions to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pid', f'{target_name}_score'])
        for post_id, prediction in zip(post_ids, predictions):
            writer.writerow([post_id, prediction[0]])

    logger.info(f"Predictions saved to '{output_csv}'")

class VideoHeartPredictor(nn.Module):
    def __init__(self, feature_sizes,feature=256):
        super(VideoHeartPredictor, self).__init__()

        # Adjusted input sizes based on feature_sizes (e.g., feature_sizes[0] = 1)
        self.fc1_1 = nn.Linear(feature_sizes[0], feature)  # For feature1, which has 1 element
        self.fc1_2 = nn.Linear(feature_sizes[1], feature)  # Adjust for the correct sizes of other features
        self.fc1_3 = nn.Linear(feature_sizes[2], feature)
        self.fc1_4 = nn.Linear(feature_sizes[3], feature)
        self.fc1_5 = nn.Linear(feature_sizes[4], feature)
        self.fc1_6 = nn.Linear(feature_sizes[5], feature)

        # Batch normalization layers for each feature
        self.bn1_1 = nn.BatchNorm1d(feature)
        self.bn1_2 = nn.BatchNorm1d(feature)
        self.bn1_3 = nn.BatchNorm1d(feature)
        self.bn1_4 = nn.BatchNorm1d(feature)
        self.bn1_5 = nn.LayerNorm(feature)
        self.bn1_6 = nn.LayerNorm(feature)

        # Dropout layer after the first layer for regularization
        self.dropout1 = nn.Dropout(0.3)  # 30% dropout rate

        # Combining layers
        self.fc_combined = nn.Linear(feature * 6, 1024)
        self.bn_combined = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.4)  # 40% dropout rate

        # Further dense layers
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 1)

        # Apply Kaiming initialization to each layer
        self._initialize_weights()

    def forward(self, feature1, feature2, feature3, feature4, feature5, feature6):
        # Process each feature through its respective layer with batch normalization
        feature1 = F.gelu(self.bn1_1(self.fc1_1(feature1.view(feature1.size(0), -1))))
        feature2 = F.gelu(self.bn1_2(self.fc1_2(feature2.view(feature2.size(0), -1))))
        feature3 = F.gelu(self.bn1_3(self.fc1_3(feature3.view(feature3.size(0), -1))))
        feature4 = F.gelu(self.bn1_4(self.fc1_4(feature4.view(feature4.size(0), -1))))
        feature5 = F.gelu(self.bn1_5(self.fc1_5(feature5.view(feature5.size(0), -1))))
        feature6 = F.gelu(self.bn1_6(self.fc1_6(feature6.view(feature6.size(0), -1))))

        # Concatenate the processed features
        combined = torch.cat([feature1, feature2, feature3, feature4, feature5, feature6], dim=1)

        # Apply dropout and batch normalization after combining features
        combined = F.relu(self.bn_combined(self.fc_combined(combined)))
        combined = self.dropout2(combined)

        # Forward through the second layer with batch normalization and dropout
        combined = F.relu(self.bn2(self.fc2(combined)))
        combined = self.dropout3(combined)

        # Final output
        output = self.fc3(combined)

        return output

    def _initialize_weights(self):
        # Kaiming initialization for the fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)