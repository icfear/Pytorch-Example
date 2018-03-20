import torch
from torch.autograd import Variable

#-----------------CASE 1--------------------------
print("-----------------CASE 1--------------------------")

x = Variable(torch.ones(2, 2), requires_grad=True)
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2), requires_grad=True)
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)


#-----------------CASE 2--------------------------
print("-----------------CASE 2--------------------------")

x = Variable(torch.ones(2, 2))
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2), requires_grad=True)
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)

#-----------------CASE 3--------------------------
print("-----------------CASE 3--------------------------")

x = Variable(torch.ones(2, 2))
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2))
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)



#-----------------CASE 4--------------------------
print("-----------------CASE 4--------------------------")

#x = Variable(torch.ones(2, 2), requires_grad=True, volatile=True)
#Returns error ->Variable can't be volatile and require_grad at the same time!
#We could manually set requires_grad parameter
x = Variable(torch.ones(2, 2), volatile=True)
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

x.requires_grad=True
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2), volatile=True)
y.requires_grad=True
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)


#-----------------CASE 5--------------------------
print("-----------------CASE 5--------------------------")

x = Variable(torch.ones(2, 2),  volatile=True)
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2), requires_grad=True)
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)


#-----------------CASE 6--------------------------
print("-----------------CASE 6--------------------------")

x = Variable(torch.ones(2, 2),  volatile=True)
print("x.requires_grad -> %r" %x.requires_grad)
print("x.volatile -> %r\n" %x.volatile)

y = Variable(torch.ones(2, 2), requires_grad=True)
print("y.requires_grad -> %r" %y.requires_grad)
print("y.volatile -> %r\n" %y.volatile)

z = x + y
print("z.requires_grad -> %r" %z.requires_grad)
print("z.volatile -> %r\n" %z.volatile)

w = Variable(z.data)
print("w.requires_grad -> %r" %w.requires_grad)
print("w.volatile -> %r\n" %w.volatile)