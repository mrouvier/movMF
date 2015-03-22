import numpy

name="a"

def next_string(s):
    strip_zs = s.rstrip('z')
    if strip_zs:
        return strip_zs[:-1] + chr(ord(strip_zs[-1]) + 1) + 'a' * (len(s) - len(strip_zs))
    else:
        return 'a' * (len(s) + 1)


mu, kappa = 1.0, 3.0
for x in range(0, 10000):
    s = numpy.random.vonmises(mu, kappa, 100)
    print name+" "+" ".join( map(str, s) )
    name=next_string(name)


mu, kappa = -1.0, 2.5
for x in range(0, 10000):
    s = numpy.random.vonmises(mu, kappa, 100)
    print name+" "+" ".join( map(str, s) )
    name=next_string(name)


mu, kappa = 3.0, 1.5
for x in range(0, 10000):
    s = numpy.random.vonmises(mu, kappa, 100)
    print name+" "+" ".join( map(str, s) )
    name=next_string(name)


mu, kappa = -13.0, 7.5
for x in range(0, 10000):
    s = numpy.random.vonmises(mu, kappa, 100)
    print name+" "+" ".join( map(str, s) )
    name=next_string(name)


mu, kappa = 0.0, 1.5
for x in range(0, 10000):
    s = numpy.random.vonmises(mu, kappa, 100)
    print name+" "+" ".join( map(str, s) )
    name=next_string(name)
