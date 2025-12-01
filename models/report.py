import subprocess
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns


"""
This class is a convenience to abstract graph creation and automatically generate pdf reports. 

Constructor:
    name: the experiment name
    dataset: the dataset used
    learning_rate: the learning_rate used
    epochs: the number of epochs trained.

save: saves the generated report to a file.

compile: compiles the generated report to a pdf.
    Requires typst.
    typst_path: 
        The path to typst binary. Default is '/opt/homebrew/bin/typst'

add_accuracy: add an accuracy statement
    accuracy: the accuracy reported.
    alt_name: default uses the experiment name

add_loss_figure: add a standard loss figure to the report
    loss: a loss vs epoch array.
    alt_name: default uses the experiment name

add_multi_loss: add multiple loss graphs to the same figure.
    loss_dict: a dictionary with key=legend label and val=loss vs epoch array
    alt_name: default uses the experiment name

add_confusion: add a confusion figure to the report.
    results: as outputted by test.py, a dict with keys 'accuracy', 'y_true', and 'y_pred'
    classes: the names to label the classes with. 
    alt_name: default uses the experiment name

add_accuracy_graph: add a graph of accuracy over many different trials.
    accuracy_list: list of accuracy values 0:1
    x_label: what to put on the x label
    alt_name: default uses the experiment name
    x_labels: if provided, alternate tick labels for the x axis

"""
class report():
    def __init__(self,
            name: str,
            dataset: str,
            learning_rate: float,
            epochs: int,
            ):
        self.name = name
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.charts = 0
        self.report = f"""= Report for {name}
== Model Info:
Data Set = {dataset} \\
Learning Rate = {learning_rate} \\
Epochs = {epochs} \\
"""

    def save(self):
        with open(f"{self.name}/{self.name}_report.typ", "w") as f:
            f.write(self.report)

    def compile(self, typst_path="/opt/homebrew/bin/typst"):
        subprocess.run([typst_path, "compile", f"{self.name}/{self.name}_report.typ"])

    def add_accuracy(self, accuracy, alt_name=None):        
        if alt_name: name = alt_name
        else: name = self.name 
        self.report += f'== {alt_name}\nAccuracy = {accuracy}'

    def add_loss_figure(self, loss, alt_name=None):         
        if alt_name: name = alt_name
        else: name = self.name   
        plt.figure(figsize=(8,5))
        plt.plot(loss, label="Average Loss per Epoch")
        plt.xlabel("Epoch Number", fontsize=12)
        plt.ylabel("Average Loss", fontsize=12)
        plt.title(f"{name} - Average Loss Per Epoch")
        flname = f"{name}_loss.png"
        plt.savefig(f"{self.name}/{flname}")
        plt.close()
        self.report += f'#figure(image("{flname}", width:90%),caption:[Loss Graph for {name} experiment])'

    def add_multi_loss(self, loss_dict, alt_name=None):
        if alt_name: name = alt_name
        else: name = self.name
        plt.figure(figsize=(10, 6))
        for label, loss in loss_dict.items():
            plt.plot(loss, label=label)
        plt.xlabel("Epoch Number", fontsize=12)
        plt.ylabel("Average Loss", fontsize=12)
        plt.title(f"{name} - Average Loss Per Epoch")
        plt.legend()
        flname = f"{name}_mloss.png"
        plt.savefig(f"{self.name}/{flname}")
        plt.close()
        self.report += f'#figure(image("{flname}", width:90%), caption: [Loss Comparison for {name} experiment])'

    def add_confusion(self, results, classes, alt_name=None):        
        if alt_name: name = alt_name
        else: name = self.name
        confusion = skm.confusion_matrix(results['y_true'], results['y_pred'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(f'Classification Accuracy = {results['accuracy']}')
        flname = f"{name}_confusion.png"
        plt.savefig(f"{self.name}/{flname}")
        plt.close()
        self.report += f'#figure(image("{flname}", width:90%),caption: [Confusion Matrix for {name} experiment])'

    def add_accuracy_graph(self, accuracy_list, x_label, alt_name=None, x_labels=None):
        if alt_name: name = alt_name
        else: name = self.name
        plt.figure(figsize=(8, 5))
        if x_labels:
            plt.plot(range(len(accuracy_list)), accuracy_list, marker='o', markersize=4, label="Accuracy")
            plt.xticks(range(len(x_labels)), x_labels)
        else:
            plt.plot(accuracy_list, marker='o', markersize=4, label="Accuracy")
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title(f"{name} Accuracy")
        plt.ylim(0, 1)  
        flname = f"{name}_accuracy"
        plt.savefig(f"{self.name}/{flname}.png")
        plt.close()
        self.report += f'#figure(image("{flname}.png", width:90%), caption: [Accuracy Graph for {name}])'
