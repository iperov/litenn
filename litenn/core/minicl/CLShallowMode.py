
class CLShallowMode:
    stack_count = 0

    def __init__(self, devices):
        self.devices = devices

    def __enter__(self):
        CLShallowMode.stack_count += 1

    def __exit__(self, a,b,c):
        CLShallowMode.stack_count -= 1
        if CLShallowMode.stack_count < 0:
            raise ValueError('Wrong stack_count.')

