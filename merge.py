import argparse
import numpy as np
import pandas as pd
import csv
import torch
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="merge final score")
    parser.add_argument('--model', type=str, default='train_val',
                        help="train_val or train_test")
    parser.add_argument('--method', type=str, default='avg',
                        help="avg,learn, jiaquan")
    parser.add_argument('--paths', type=str, default='',
                        help="paths of excel and video methods, separated by commas")
    args = parser.parse_args()

    paths_all = args.paths.split(',')
    
    results= {}
    for path in paths_all:
        single_result = pd.read_csv(path)
        for pid, pred in zip(single_result['pid'], single_result['popularity_score']):
            if pid not in results:
                results[pid] = []
            results[pid].append(pred)

    final_result = {}

    if args.model == 'train_test':
        if args.method == 'avg':
            final_result = {pid: np.mean(preds) for pid, preds in results.items()}

        with open(f'./merge_result.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pid', f'popularity_score'])
            sorted_results = sorted(final_result.items(), key=lambda x: x[0])
            for post_id, prediction in sorted_results:
                writer.writerow([post_id, round(prediction,2)])
    elif args.model == 'train_val':
        # Assume we have a ground truth file for training
        ground_truth_path = input("Please input the path of the ground truth csv file: ")
        ground_truth = pd.read_csv(ground_truth_path)

        if args.method == 'learn':
            # Convert results to a DataFrame
            data = []
            for pid, preds in results.items():
                data.append([pid] + preds)
            df = pd.DataFrame(data, columns=['pid'] + [f'model_{i}' for i in range(len(paths_all))])
        
            # Merge ground truth with predictions
            merged_df = pd.merge(df, ground_truth, on='pid', how='inner')
            
            # Prepare data for training
            X = merged_df[[f'model_{i}' for i in range(len(paths_all))]].values
            y = merged_df['popularity'].values
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Train linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            print(f"Validation mape: {mape}")
        
            # Extract model parameters
            weights = model.coef_
            intercept = model.intercept_

            print(f"Weights: {weights}", f"Intercept: {intercept}")
            # Convert to PyTorch tensors
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            intercept_tensor = torch.tensor(intercept, dtype=torch.float32)

            # Save parameters to .pt file
            torch.save({'weights': weights_tensor, 'intercept': intercept_tensor}, 'linear_regression_model.pt')
            print("Model parameters saved to linear_regression_model.pt")
        elif args.method == 'avg':
            final_result = {pid: np.mean(preds) for pid, preds in results.items()}
            mape = mean_absolute_percentage_error(ground_truth['popularity'], list(final_result.values()))
            print(f"Validation mape: {mape}")
        elif args.method == 'jiaquan' and len(paths_all) == 2:
            mape_min = 10000
            for i in range(0,11):
                for pid, preds in results.items():
                    final_result[pid] = preds[0] * i / 10 + preds[1] * (10 - i) / 10
                mape = mean_absolute_percentage_error(ground_truth['popularity'], list(final_result.values()))
                if mape < mape_min:
                    mape_min = mape
                    best_i = i
                print(i, mape)
            print(f"Best i: {best_i}, MAPE: {mape_min}")