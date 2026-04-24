# Install the new package rather than using the old "keras" package. 
# install.packages("keras3")
# install.packages("torch")

# Load the pROC library
# install.packages("pROC")
library(pROC)

# Initialize Keras (This ensures the Python backend is linked correctly)
library(keras3)
# install_keras() or specifically 
# install_keras(backend = "tensorflow")

# Using Neural Network (Keras): The "Hidden Layers" (layer_dense) act as a Black Box. 
# They create complex, non-linear combinations of variables (like combining age, income, and housing_type 
# into a single abstract "score") to find patterns that a simple decision tree might miss.

# 0. Load any required dependency libraries. 
# Maybe I might need to ask students using 4.2.x version to check reticulate ---- 
library(dplyr)
library(tidyr)

# 1. Load and Pre-process Data ----
hr <- read.csv("11_HR_comma_sep.csv")
# hr$left <- ifelse(hr$left == 1, "left", "not_left")

# Note: Keras needs the target to be numeric (1 for left, 0 for not_left)
y_label <- hr$left

# Remove the target from the input features
hr_features <- hr %>% select(-left)

# 2. Transform Categorical Data (One-Hot Encoding) ----
# The HR data has 'sales' (department) and 'salary' (low, medium, high).
# model.matrix creates the necessary dummy variables for these strings.
hr_numeric <- model.matrix(~ . - 1, data = hr_features)

# 3. Scaling (Min-Max Normalization) ----
# Essential for Neural Networks to handle satisfaction_level (0-1) vs. average_montly_hours (100-300)
maxs <- apply(hr_numeric, 2, max)
mins <- apply(hr_numeric, 2, min)
hr_scaled <- as.data.frame(scale(hr_numeric, center = mins, scale = maxs - mins))

# 4. Train/Test Split ----
set.seed(1234)
index <- sample(c("train", "test"), nrow(hr_scaled), replace = TRUE, prob = c(0.8, 0.2))

x_train <- as.matrix(hr_scaled[index == "train", ])
y_train <- as.matrix(y_label[index == "train"])
x_test  <- as.matrix(hr_scaled[index == "test", ])
y_test  <- as.matrix(y_label[index == "test"])

# 5. Build Model Architecture ----
# We use ReLU for hidden layers and Sigmoid for the final binary output.
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid") 

# 6. Compile and Fit ----
model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy", 
  metrics = c("accuracy")
)

# Optional: Adjust weights if the classes are imbalanced
# In this data, 'not_left' is 0 and 'left' is 1. 
# If 'left' cases are fewer, increase the weight for "1".
weights <- list("0" = 1.0, "1" = 2.0) 

history <- model %>% fit(
  x_train, y_train,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
  #class_weight = weights,
  verbose = 1
)

# 7. Predict and Threshold Optimization (ROC) ----
predictions_prob <- model %>% predict(x_test)
head(predictions_prob)
# Below code is for picking the right threshold using one of the available method (Youden). Experiment with others too.

# Create ROC object to find the best cut-off
roc_obj <- roc(as.vector(y_test), as.vector(predictions_prob))


# Plot the ROC Curve for students
plot(roc_obj, main ="ROC Curve: HR Employee Turnover", col = "#e67e22", lwd = 4, print.auc = TRUE)

# Apply the optimal threshold. Find this "Best" threshold using Youden's J statistic
# This maximizes (Sensitivity + Specificity - 1)
best_threshold <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
best_threshold <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
print(paste("The optimal threshold is:", best_threshold))

predictions_final <- ifelse(predictions_prob > best_threshold$threshold, 1, 0)

# 8. Evaluation ----
table(Actual = y_test, Predicted = predictions_final)
