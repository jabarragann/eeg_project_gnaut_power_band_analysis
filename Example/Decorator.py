def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner

@make_pretty
def ordinary():
    print("I am ordinary")


class test:

    def __init__(self, m):
        self.aOrb=m

    def aMethod(self):
        print("Do a methods")

    def bMethod(self):
        print("Do b methods")

    def procedure(self):
        if self.aOrb == 'a':
            self.aMethod()
        elif self.aOrb == 'b':
            self.bMethod()
        else:
            raise Exception("Should be a or b")


# ordinary()
#
# pretty = make_pretty(ordinary)
# pretty()

classA = test('a')
classB = test('c')

classA.procedure()
classB.procedure()