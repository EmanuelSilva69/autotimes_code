from efficient_kan import KAN
import inspect
print(inspect.signature(KAN.__init__))
print()
# Check source for base_activation
src = inspect.getsource(KAN.__init__)
for line in src.split('\n'):
    if 'activation' in line.lower() or 'base' in line.lower():
        print(line)
