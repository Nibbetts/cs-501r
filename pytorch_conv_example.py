import torch
from torch import nn
from torch.autograd import Variable

data = Variable(10,3,256,256) # images with batch size of 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # in_channels, out_channels, kernel_size (filter size):
        conv1 = nn.Conv2d(3,6,3,stride=1)
        conv2 = nn.Conv2d(6,12,3,stride=1)
        conv3 = nn.Conv2d(12,18,3,stride=1)
        # use shift+tab in jupyter notebooks to get docstrings to see what goes into a function

        conv4 = nn.Conv2d(18,24,3,stride=2)
        conv5 = nn.Conv2d(24,36,3,stride=1)
        conv6 = nn.Conv2d(36,48,3,stride=1)

        conv7 = nn.Conv2d(48,64,3,stride=2)
        conv8 = nn.Conv2d(64,64,3,stride=1)
        conv9 = nn.Conv2d(64,64,3,stride=1)

        self.conv_units = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8,self.conv9)
        self.fc = nn.Linear(64*14*14,10)

        #if resnet:
        #self.block1 = nn.Sequential(self.conv1, self.conv2, self.conv3)
        #...
        #...

    def forward(self, input_):
        #if resnet:
        #output1 = self.conv1(input_)
        #output2 = output1 + self.conv2(output1)

        #output3_ = self.conv3(output2)
        #output3 = output3_ + output2

        #output4 = self.conv4....
        #...

        #watch the strides! don't want to add tensors of different shape!
        #

        return self.fc(self.conv_units(input_).view(17,-1))


# batch, channels, height, width:
data = Variable(torch.rand(17,3,256,256))
ground_truth = Variable(torch.LongTensor([3,5,7,3,5,3,6,7,3,2,5,5,4,3,5,7,3]) #this should be changed out for real data.
#note, don't have to use one-hots with pytorch!

model = Classifier()

loss_op = nn.CrossEntropyLoss()
optim = torch.optim.Adam()

for i in range(n_epochs):
    for j in range(n_minibatches):

        #data_in = get_batch(17)
        data_in = data
        #ground_truth =

        #forward pass
        result = model(data)
        loss = loss_op(result, ground_truth)

        #backward pass
        loss.backward()

        #optimization
        optim.step()
        optim.zero_grad() #don't want to keep accumulating gradients! reset after eachpyt

#type(loss)
#result #to test/see
#result.size() #To test as we go along

#DEMONSTRATION of how to get the first parameter:
#first_parameter = list(conv1.parameters())[0] # conv1.parameters() returns a generator, which advances with .next
#fisrt_parameter.data.sub_(first_parameter.grad.data*.01)
#first_parameter.grad
#OR FOR ALL:
for first_parameter in conv1.parameters():
    first_parameter.data.sub_(first_parameter.grad.data*.01)
    first_parameter.grad

#result.transpose() #to rearrange
