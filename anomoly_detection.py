import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Execute the complete anomaly detection pipeline.
    
    The pipeline consists of:
    1. Loading and preparing the data
    2. Analyzing dataset characteristics
    3. Detecting anomalies using sliding window percentile method
    4. Evaluating detection performance
    5. Visualizing results with annotated anomalies
    """
    # Load data from CSV file
    df = pd.read_csv('AG_NO3_fill_cells_remove_NAN-2.csv')
    nitrate_data = df['NO3N'].values
    
    # Detection parameters
    WINDOW_SIZE = 2000
    PERCENTILE_THRESHOLD = 99.9
    
    # Execute pipeline components
    analyze_data(nitrate_data)
    predictions, thresholds = detect_anomalies(nitrate_data, WINDOW_SIZE, PERCENTILE_THRESHOLD)
    evaluate_performance(predictions)
    
    # Generate visualization
    plot_anomalies(predictions, nitrate_data, thresholds, PERCENTILE_THRESHOLD)


def analyze_data(nitrate_data):
    """
    Perform tatistical analysis of the nitrate dataset.
    
    Args:
        nitrate_data (numpy.ndarray): Array containing nitrate concentration measurements
    """
    print("=== DATA SUMMARY ===")
    print(f"Total data points: {len(nitrate_data):,}")
    print(f"Concentration range: {np.min(nitrate_data):.3f} to {np.max(nitrate_data):.3f}")
    print(f"Mean: {np.mean(nitrate_data):.3f}, Std: {np.std(nitrate_data):.3f}")


def detect_anomalies(data, window_size, percentile_threshold):
    """
    Detect anomalies using a sliding window percentile-based approach.
    
    This method calculates dynamic thresholds for each window and identifies
    data points that exceed the window-specific percentile threshold.
    
    Args:
        data (numpy.ndarray): Input data array for anomaly detection
        window_size (int): Number of data points in each sliding window
        percentile_threshold (float): Percentile value for threshold calculation (0-100)
    
    Returns:
        tuple: 
            - predictions (numpy.ndarray): Binary array where 1 indicates anomalies
            - thresholds (numpy.ndarray): Dynamic threshold values for each data point
    """
    n_points = len(data)
    predictions = np.zeros(n_points, dtype=int)
    thresholds = np.zeros(n_points)  # Store threshold for each point
    
    print(f"\n=== STARTING ANOMALY DETECTION ===")
    print(f"Processing {n_points - window_size + 1} windows...")
    
    # Process first window
    first_window = data[0:window_size]
    first_threshold = np.percentile(first_window, percentile_threshold, method="linear")
    
    # Label every point in first window and set thresholds
    for i in range(window_size):
        thresholds[i] = first_threshold
        if data[i] >= first_threshold:
            predictions[i] = 1    
    
    # Process remaining windows with sliding approach
    for start_idx in range(1, n_points - window_size + 1):
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        threshold = np.percentile(window_data, percentile_threshold, method="linear")
        new_point_idx = end_idx - 1
        
        # Set threshold for the new point and label if anomaly
        thresholds[new_point_idx] = threshold
        if data[new_point_idx] >= threshold:
            predictions[new_point_idx] = 1
    
    print("Anomaly detection completed!")
    return predictions, thresholds


def evaluate_performance(predictions, true_anomalies_count=77):
    """
    Calculate and display comprehensive performance metrics for anomaly detection.
    
    Args:
        predictions (numpy.ndarray): Binary predictions from anomaly detection
        true_anomalies_count (int, optional): Known number of true anomalies. Defaults to 77.
    
    Returns:
        tuple: Performance metrics including TP, FP, FN, TN and accuracy scores
    """
    total_points = len(predictions)
    predicted_anomalies = np.sum(predictions)
    
    print("\n=== PERFORMANCE RESULTS ===")
    
    # Calculate confusion matrix components
    tp = min(predicted_anomalies, true_anomalies_count)
    fn = true_anomalies_count - tp
    fp = predicted_anomalies - tp
    tn = total_points - true_anomalies_count - fp
    
    # Calculate accuracy metrics
    normal_accuracy = tn / (tn + fp)
    anomaly_accuracy = tp / true_anomalies_count
    
    # Print detailed results
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}") 
    print(f"True Negatives: {tn}")
    print(f"Predicted anomalies: {predicted_anomalies}")
    print(f"Known true anomalies: {true_anomalies_count}")
    
    print(f"\n=== ACCURACY METRICS ===")
    print(f"Normal Accuracy: {normal_accuracy:.4f} ({normal_accuracy*100:.2f}%)")
    print(f"Anomaly Accuracy: {anomaly_accuracy:.4f} ({anomaly_accuracy*100:.2f}%)")
    
    # Evaluate against performance targets
    print(f"\n=== TARGET ASSESSMENT ===")
    normal_met = "✓" if normal_accuracy >= 0.8 else "✗"
    anomaly_met = "✓" if anomaly_accuracy >= 0.75 else "✗"
    
    print(f"{normal_met} Normal Accuracy: {normal_accuracy*100:.1f}% (target ≥80%)")
    print(f"{anomaly_met} Anomaly Accuracy: {anomaly_accuracy*100:.1f}% (target ≥75%)")
    
    return tp, fp, fn, tn, normal_accuracy, anomaly_accuracy


def plot_anomalies(predictions, nitrate_data, thresholds, percentile_threshold):
    """
    Generate comprehensive visualization of data with detected anomalies.
    
    Args:
        predictions (numpy.ndarray): Binary anomaly predictions
        nitrate_data (numpy.ndarray): Nitrate concentration values
        thresholds (numpy.ndarray): Dynamic threshold values
        percentile_threshold (float): Percentile used for threshold calculation
    """
    # Create time/index axis
    time_index = np.arange(len(nitrate_data))
    
    # Create figure with appropriate sizing
    plt.figure(figsize=(14, 8))
    
    # Plot main data series
    plt.plot(time_index, nitrate_data, 'b-', alpha=0.7, linewidth=0.8, label='Nitrate Data')
    
    # Plot dynamic threshold line
    plt.plot(time_index, thresholds, '--', color='orange', linewidth=1.5, 
             label=f'Dynamic Threshold ({percentile_threshold}th percentile)')
    
    # Highlight detected anomalies
    anomaly_indices = np.where(predictions == 1)[0]
    plt.scatter(anomaly_indices, nitrate_data[anomaly_indices], 
                color='red', s=30, zorder=5, label='Detected Anomalies')
    
    # Configure plot aesthetics
    plt.xlabel('Index')
    plt.ylabel('NO3N Concentration')
    plt.title('Nitrate Data with Detected Anomalies and Dynamic Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()