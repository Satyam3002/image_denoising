class A:
    def __init__(self):
        d = 10

    def forward(self):
        print(self.d)


if __name__ == "__main__":
    obj = A()
    obj.forward()
