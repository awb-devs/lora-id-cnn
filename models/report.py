import subprocess
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns


def generate_report(
        name: str,
        dataset: str,
        learning_rate: int,
        epochs: int,
        note: str,
        loss,
        results,
        classes
        ):
    # training loss
    plt.figure(figsize=(8,5))
    plt.plot(loss, label="Average Loss per Epoch")
    plt.xlabel("Epoch Number", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.title(f"{name} Average Loss Per Epoch")
    plt.savefig(f"{name}/{name}-training_loss.png")
    plt.close()
    # confusion matrix
    confusion = skm.confusion_matrix(results['y_true'], results['y_pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {name}')
    plt.savefig(f"{name}/{name}-confusion.png")
    plt.close()
    # pdf report
    report = f"""= Report for {name}
Accuracy = {results['accuracy']}
== Model Info:
Model Name = TODO \\
Data Set = {dataset} \\
Learning Rate = {learning_rate} \\
Epochs = {epochs} \\
Notes = {note} \\
== Training Loss
#figure(
    image("{name}-training_loss.png")
)
== Confusion Matrix
#figure(
    image("{name}-confusion.png")
)
"""
    # save report
    with open(f"{name}/{name}-report.typ", "w") as f:
        f.write(report);
    subprocess.run(["/opt/homebrew/bin/typst", "compile", f"{name}/{name}-report.typ"])





