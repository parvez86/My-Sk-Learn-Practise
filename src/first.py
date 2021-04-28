import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i**2 for i in range(10)]

# plt.plot(x,y)
plt.scatter(x,y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()