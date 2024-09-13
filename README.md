## Task Description

The problem involves predicting the future movement of traffic agents for autonomous vehicles, where challenges include:
- Unpredictable human behavior, obstacles, and weather conditions
- Continuously changing environments, requiring adaptable models

The model aims to enhance the safety and efficiency of autonomous vehicles by improving motion forecasting, potentially saving lives and reducing accidents.

---

## Dataset and Preprocessing

### Data Statistics:
- **Training set**: 205,942 samples
- **Test set**: 3,200 samples
- **Input dimensions**: `[60, 19, 4]` (position and velocity data for 60 agents over 19 time steps)
- **Output dimensions**: `[60, 30, 4]` (position and velocity data over 30 time steps)

### Preprocessing Techniques:
1. **Initial Preprocessing**:
   - Extracted position and velocity data for agents.
   - Optimized data conversion to improve training speed using `numpy.stack` for list to tensor conversion.

2. **Agent Filtering**:
   - Focused on active agents, filtered inactive agents, reducing data dimensionality and focusing on relevant features.

3. **Data Normalization**:
   - Applied translation (setting the first point as origin) and rotation (aligning the 19th point along the x-axis) to standardize data across samples, improving model training and accuracy.

---

## Models and Architecture

We implemented several deep learning models to solve the motion forecasting problem:

### 1. CNN (Convolutional Neural Network)
- **Architecture**: A 3-layer CNN with ReLU activation.
- **Input**: A tensor of size `[60, 19, 4]` (position and velocity of 60 agents over 19 time steps).
- **Output**: Predicted positions over 30 time steps.
- **Performance**: The CNN model showed mixed performance, with improvements needed in the handling of sparse input data.

### 2. LSTM (Long Short-Term Memory)
- **Architecture**: 2 LSTM layers with 64 and 32 hidden units.
- **Input**: A 3D tensor of features from 19 time stamps, capturing the temporal aspect of the data.
- **Performance**: The LSTM model underperformed due to short time dependencies in the data.

### 3. MLP (Multi-Layer Perceptron)
- **Architecture**: A simple 3-layer MLP with output units of 64, 16, and 4.
- **Performance**: The MLP model outperformed the CNN and LSTM models, particularly in cases of simple directional movement. However, it struggled with scenarios involving stopping vehicles.

### 4. CLDNN (Convolutional LSTM Deep Neural Network)
- **Best Performing Model**: CLDNN version 4
- **Architecture**: A combination of CNN for spatial feature extraction and LSTM for temporal prediction, with manual feature extraction added for improved prediction.
- **Performance**: Achieved the best results with an RMSE of 1.78 on the Kaggle public test set.

---

## Experiment Design and Results

### Experiment Workflow:
- Regular team meetings to discuss progress and share findings.
- Training environments: Google Colab and local machines with Nvidia RTX 3080ti GPU.
- Optimization: Models trained using Adam and SGD optimizers, with early stopping applied based on validation loss.
- The final weight for each model was selected based on the lowest validation loss.

### Model Comparison:
| Model         | Time per Epoch (s) | RMSE on Kaggle Test Set |
|---------------|--------------------|-------------------------|
| CNN           | 150                | 47.02                   |
| LSTM          | 43                 | 127.25                  |
| MLP (3-layer) | 26                 | 2.12                    |
| MLP (4-layer) | 28                 | 2.35                    |
| CLDNN v4      | 450                | 1.78                    |

The CLDNN v4 model significantly outperformed other models in terms of RMSE, making it the best solution for the task. However, it required more computational resources due to the model's complexity.

---

## Key Insights and Future Work

### Key Insights:
- **Preprocessing**: Translational and rotational normalization of the data improved model accuracy. These transformations helped align input features, making patterns easier to learn for the models.
- **Model Complexity**: While simpler models like MLP were faster to train, the complex CLDNN architecture with both spatial and temporal feature extraction yielded the best performance.
- **Prediction Strategy**: Predicting 30 frames together rather than one frame at a time reduced error accumulation, significantly improving overall prediction accuracy.

### Future Improvements:
- **Incorporating Neighboring Agents**: Future work will involve utilizing data from neighboring agents to predict trajectories, factoring in interactions between vehicles in the traffic environment.
- **Graph Neural Networks (GNN)**: We plan to explore GNNs to model the spatial relationships between agents.
- **Attention Mechanisms**: Adding attention mechanisms to the CLDNN model could help the network focus on important agents or lane markers when predicting future trajectories.

---

