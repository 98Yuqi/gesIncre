from increment import *
from cvae_increment import *
from data import *
from model import *
from model2 import *

num_task = 8
num_class = 2
epochs1 = 1000
epochs2 = 500
data_total, label_total = load_data("4")
train_loader,test_loader = getDataLoader(data_total, label_total, num_class)

model1 = classifier()
model2 = antoencoder()
# cuda_available = False
# if torch.cuda.is_available():
#     cuda_available = True

memsize_acc = []
gem = increLearning(model1, num_task)
cvae = CVAELearning(model2)

for i in range(num_task):
    memory, memory_label = cvae.before(i)
    for epoch in range(epochs1):
        cvae.train(train_loader[i], epoch, memory, memory_label)
    for epoch in range(epochs2):
        gem.train(train_loader[i], i, epoch)
    for j in range(i + 1):
        gem.eval(test_loader[j],j,i)
    print(gem.R)
    memsize_acc.append(torch.sum(gem.R[i]).item() / (i + 1))
    recon, recon_label = cvae.eval(i)
    gem.after(i,recon, recon_label)
