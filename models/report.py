import subprocess
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns


class report():
    def __init__(self,
            name: str,
            dataset: str,
            learning_rate: float,
            epochs: int,
            accuracy,
            classes
            ):
        self.name = name
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accuracy = accuracy
        self.classes = classes
        self.report = f"""= Report for {name}
Accuracy = {accuracy}
== Model Info:
Data Set = {dataset} \\
Learning Rate = {learning_rate} \\
Epochs = {epochs} \\
"""

    def save(self):
        with open(f"{self.name}/{self.name}_report.typ", "w") as f:
            f.write(self.report)

    def compile(self):
        subprocess.run(["/opt/homebrew/bin/typst", "compile", f"{self.name}/{self.name}_report.typ"])

    def add_loss_figure(self, loss):    
        name = self.name
        plt.figure(figsize=(8,5))
        plt.plot(loss, label="Average Loss per Epoch")
        plt.xlabel("Epoch Number", fontsize=12)
        plt.ylabel("Average Loss", fontsize=12)
        plt.title(f"{name} Average Loss Per Epoch")
        plt.savefig(f"{name}/{name}_loss.png")
        plt.close()
        self.report += f'#figure(image("{name}_loss.png", width:90%),caption:[Loss Graph for {name} experiment])'

    def add_confusion(self, results):
        name = self.name
        classes = self.classes
        confusion = skm.confusion_matrix(results['y_true'], results['y_pred'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(f'Classification Accuracy = {self.accuracy}')
        plt.savefig(f"{name}/{name}_confusion.png")
        plt.close()
        self.report += f'#figure(image("{name}_confusion.png", width:90%),caption: [Confusion Matrix for {name} experiment])'
