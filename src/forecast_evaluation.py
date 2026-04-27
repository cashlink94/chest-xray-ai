from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'y_true' are the true labels and 'y_pred' are predictions from the model
y_pred = model.predict(val_data)  # Predictions on validation data
y_true = val_data.classes  # Actual labels

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, np.argmax(y_pred, axis=1))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_data.class_indices, yticklabels=val_data.class_indices)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# Save the confusion matrix plot
confusion_matrix_path = 'outputs/plots/confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()

print(f"Confusion matrix saved: {confusion_matrix_path}")