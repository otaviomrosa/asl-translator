from sklearn.metrics import confusion_matrix, classification_report
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import SignLanguageDataset
from src.model import ASLClassifier
from torch.utils.data import DataLoader
import seaborn as sns
import os


F = os.path.dirname(os.path.abspath(__file__))

# Letters
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H",
           "I", "K", "L", "M", "N", "O", "P",
           "Q", "R", "S", "T", "U", "V", "W", "X", "Y",]


def load(csv: str = "data/sign_mnist_test.csv"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    model_p = 'models/asl_model.pth'
    batch = 64
    model = ASLClassifier().to(device)
    m_load = torch.load(model_p, map_location=device)
    model.load_state_dict(m_load)
    model.eval()


    # loading data set and test loader
    test_set = SignLanguageDataset(csv)
    test_load = DataLoader(test_set, batch_size=batch, shuffle = False)

    labels_list = []
    predict_list = []
    r_list = []

    with torch.no_grad():
        for img, label in test_load:
            img = img.float()
            img = img.unsqueeze(1) # adding another channel
            img = img.to(device)
            label = label.to(device)

            # getting predictions
            raw_num = model(img)
            prob_num = torch.softmax(raw_num, dim=1)
            out, idx = prob_num.max(dim=1)

            predict_list.append(idx.cpu())
            labels_list.append(label.cpu())
            r_list.append(out.cpu())


        pr = torch.cat(predict_list).numpy()
        plc = torch.cat(labels_list).numpy()
        plrt = torch.cat(r_list).numpy()

    return pr, plc, plrt


# confusion matrix heatmap
def conf_matrix(valid, pred):

    confusion = confusion_matrix(valid, pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion,annot=True, cmap= 'coolwarm', fmt='d', xticklabels=LETTERS, yticklabels=LETTERS)

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plot_f = os.path.join(F, "cm_plot.png")
    plt.tight_layout()
    plt.savefig(plot_f)
    print("Saved Confusion Matrix")
    #plt.show()


    return confusion


def analysis(conf, pr, plc, plrt, report):
    print("Analysis starting")
    c_pairs = []

    # confused pairs
    print("Top 10 Confused pairs:")
    for i in range(24): # i = row index (y-axis) --> True Label
        for j in range(24): # j = column index (x-axis) --> Predicted Label
            if i == j: # continue if prediction is correct
                continue
            output = conf[i,j]
            if output > 0:
                c_pairs.append((LETTERS[i], LETTERS[j], output))
    c_pairs.sort(key=lambda x: x[2], reverse=True)

    out = 1
    for true, pre, num in c_pairs[:10]:
        count = int(num)
        print(f"{out}-->  Predicted Label: '{pre}', True Label: '{true}', and Occurences:{count} ")
        out +=1


    # per class accuracy
    c_count = conf.sum(axis=1) 
    c_correct = np.diag(conf)

    class_acc = c_correct / c_count
    sort = np.argsort(class_acc)

    # highest accuracy
    high = sort[-3:][::-1]
    # lowest accuracy
    low = sort[:3]

    print(f"Highest Accuracy Letters:")
    for i in high:
        print(f"{LETTERS[i]}: {class_acc[i]:.3f} ")

    print(f"Lowest Accuracy Letters:")
    for i in low:
        print(f"{LETTERS[i]}: {class_acc[i]:.3f} ")


    # overall accuracy
    ct= np.trace(conf)
    sum_v = conf.sum()
    if sum_v > 0:
        acc = ct/sum_v
    else:
        acc = 0.0


    out_result = os.path.join(F, "analysis.txt")

    with open(out_result, "w") as f: # writing to a txt file
        f.write("           ASL Analysis Report\n")
        f.write("\n")

        f.write("       === Top 10 Confused Letter Pairs (not diagonal): ===")
        f.write("\n")
        f.write("\n")

        out = 1
        for true, pre, num in c_pairs[:10]:
            count = int(num)
            f.write(f"{out}-->  Predicted Label: '{pre}', True Label: '{true}', and Occurences:{count}\n ")
            out +=1
        f.write("\n")

        f.write("Per Class Accuracy:\n")
        for x,y in enumerate(LETTERS):
            c_count = conf.sum(axis=1) 
            c_correct = np.diag(conf)
            class_acc = c_correct / c_count
            sort = np.argsort(class_acc)
            f.write(f"{y}: {class_acc[x]:.4f}\n")
        f.write("\n")


        print(f"Highest Accuracy Letters:")
        for i in high:
            print(f"{LETTERS[i]}: {class_acc[i]:.3f}\n ")

        print(f"Lowest Accuracy Letters:")
        for i in low:
            print(f"{LETTERS[i]}: {class_acc[i]:.3f}\n ")

        print(f"Overall Accuracy: {acc:.4f}\n")

        f.write("       ==== Classification Report:==== \n\n")
        f.write(report)


    print("Saved Report")


    return acc


def main():
    
    pr, lab, c_cn = load()
    report = classification_report(lab, pr, target_names=LETTERS)
    conf = conf_matrix(lab, pr)
    accuracy = analysis(conf, pr, lab, c_cn, report)
    




if __name__ == "__main__":
    main()



