from opticalsystem import OpSystem

# Instantiate a class object.

system1 = OpSystem()

# Plot ray diagram of optical system system1.
system1.ray_diagram()

# Plot spot diagram of bundle at screen.
system1.spot_diagram()

# Output rms spread of bundle at screen.
print(system1.rms())
