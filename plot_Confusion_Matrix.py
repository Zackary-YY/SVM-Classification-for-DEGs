import seaborn as sns
# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "RA"], yticklabels=["Normal", "RA"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Split Ratio 0.5)")
# plt.savefig(f"{dir_svm}/svm_confusion_matrix_split_0.5.pdf")
plt.show()
