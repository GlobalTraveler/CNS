from pylab import *
from numpy import *

random.seed(1234)
nActions = 8


trialDuration =  120 # sec
dt            = .1
simSteps      = int(trialDuration / dt)

nTrials       = 25

# setup the maze
mazeAngles = arange(0, 1, .01) * 2 * pi
mazeCoordinates = vstack((cos(mazeAngles), sin(mazeAngles)))

# setup target
targetWidth     = .05
targetRadius    = .5
targetAngle     = 5/4 * pi
targetCenter    = vstack((cos(targetAngle), sin(targetAngle))) * targetRadius
targetBoundaries = vstack((cos(mazeAngles), sin(mazeAngles)))*targetWidth + targetCenter

# setup placefields neurons
nNeurons = 493 # approximate with sqrt
nNeuronsSimply = floor(sqrt(nNeurons))
sigma    = .16 # m
x, y = list(random.rand(nNeurons, 2).T * 2 * pi)
placeNeurons = vstack( (cos(x), sin(y) ) )


def firingRate(pos, prefPos = placeNeurons):
    firingRates = zeros( ( len(pos.T), len(prefPos.T) ) )
    # print(firingRates.shape)
    for idx, p in enumerate(pos.T):
        for jdx, pp in enumerate(prefPos.T):
            d = array(p-pp, ndmin = 2)
            firingRates[idx, jdx] = exp(-d.dot(d.T)/(2*sigma**2))[0]
    return firingRates

fr = firingRate(mazeCoordinates)

# rat
vRat = .3 # m/s
ratPos = zeros((nTrials, simSteps, 2)) # x, y
mRat = .9

# rat - actor
ratAngles = zeros( (  nTrials, simSteps ) )
ratPos    = zeros( ( nTrials, simSteps, 2 ))
epsilon = .1
beta = 2
u = zeros( ( nActions, nTrials ) )
v = zeros(u.shape)

# rat - critic
gamma = .9975
w = zeros( ( nNeurons, 1) )
c = zeros( ( nTrials, nNeurons) )
z = zeros( ( nActions, nNeurons) )
C = zeros( ( nTrials, len(mazeCoordinates.T) ) ) # change this

uu = zeros(( nTrials, nNeurons, 2))
close('all')
fig, ax = subplots()
[dd] = ax.plot(0,0, '.', markersize = 10)
ax.plot(*targetCenter,'.', markersize = 100)
ax.set_xlim([-2,2]); ax.set_ylim([-2,2])
show(0)
theta = arange(0,nActions) / nActions * 2 * pi

def wrapToPi(x):
    x = x - floor(x / 2 * pi) * 2 * pi - 2 * pi
    return(x)
for trial in range(nTrials):
    goal = 0
    t = pi / 2
    start = vstack((cos(t), sin(t)))
    ratAngles[trial, 0] = wrapToPi(t + pi)
    print(start)
    for ti in range(simSteps):
        ratPos[trial, ti, :] = start.flatten()
        # acting
        fr = firingRate(start)
        tmp = exp(fr.dot(z.T)) * beta
        # assert 0
        p  = tmp / sum(tmp)
        # new action; weighted by experience
        j = random.choice(nActions, 1, 1, p.flatten())
        # update angle plus momentum
        tj = ratAngles[trial, ti] + ( 1- mRat) * theta[j]
        # wrap back to [0, 2pi]
        tj -= floor(tj / (2*pi))*2 *pi - 2 * pi
        tryMove = vstack( ([cos(tj)], [sin(tj)] ) ) * dt * vRat
        # print(start, tj)
        ratPos[trial + 1, ti, :] = start.flatten()
        if linalg.norm(start + tryMove) >= 1:
            print('b', tj)
            tj  -= pi
            print('a', tj);
            print(j)
            tj -= floor(tj / (2*pi))*2 *pi - 2 * pi
        start += vstack(([cos(tj)], [sin(tj)])) * dt * vRat
        dd.set_data(*start)
        pause(.001)

        frNext = firingRate(start)
        c = fr.dot(w)
        cNext = frNext.dot(w)
        ratAngles[trial, ti] = tj
        if norm(start - targetCenter) <= targetRadius:
            reward = 1; goal =  1
        else: reward = 0
        delta = reward + gamma * cNext - c
        update = fr.T.dot(delta) * epsilon
        w += update
        z[j, :] += update.flatten()
        if goal == 1:
            break
    # update the grid
    fr = firingRate(mazeCoordinates)
    updateC      = (fr.dot(w)).flatten()
    C[trial, :]  = updateC / max(updateC)
    z /= np.max(z)
    idx = array(np.max(z, 0), ndmin = 2)
    u = hstack( (cos(idx).T, sin(idx).T) )
    uu[trial, :] = u
