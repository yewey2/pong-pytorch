import random, time

RENDERING = False

screenwidth,screenheight = (300,300)
ballwidth,ballheight=(10,10)
playerwidth,playerheight=(100,20)
balllist=list()
walllist=list()
objlist=list()

ball=None

win=None

#Pygame Stuff

import pygame
if RENDERING:
    pygame.init()
    win=pygame.display.set_mode((screenwidth,screenheight))
    pygame.display.set_caption('Pong')

def overlap(pos1,pos2):
    #pos = x,y,width,height
    left  = max(pos1[0], pos2[0])
    top   = max(pos1[1], pos2[1])
    right = min(pos1[0]+pos1[2], pos2[0]+pos2[2])
    bot   = min(pos1[1]+pos1[3], pos2[1]+pos2[3])
    if right-left > 0 and bot-top > 0:
        area = (left-right) * (top-bot)
    else:
        area = 0
    return area
 
class Ball():
    def __init__(self, x=0,y=0,width=ballwidth,height=ballheight,xvel=10,yvel=10):
        self.x=random.randint(0,screenwidth)
        self.y=random.randint(0,screenheight//2)
        #self.x,self.y=0,0
        self.width=width
        self.height=height
        self.xvel=xvel*random.choice([-1,1])
        self.yvel=yvel

        self.pos=(self.x, self.y, self.width, self.height) 
        self.dead=False
        balllist.append(self)

    def collisionx(self,direction):
        collide = False
        temp_x = self.x + direction
        temp_hitbox = temp_x, self.y, self.width, self.height
        for wall in walllist:
            if overlap(wall.pos,temp_hitbox) > 0:
                collide = True
        return collide

    def collisiony(self,direction):
        collide = False
        temp_y = self.y + direction
        temp_hitbox = self.x, temp_y, self.width, self.height
        for wall in walllist:
            if overlap(wall.pos,temp_hitbox) > 0:
                collide = True
        return collide

    def checkdeath(self):
        if self.y+self.height >  screenwidth:
            self.dead = True

    def movement(self):
        self.checkdeath()
        if self.collisionx(self.xvel):
            self.xvel*=-1
        if self.collisiony(self.yvel):
            self.yvel*=-1
        self.x+=self.xvel
        self.y+=self.yvel
        self.updatepos()

    def updatepos(self):
        self.pos = self.x, self.y, self.width, self.height
        
        

class Player():
    def __init__(self,x=screenwidth//2-playerwidth//2,y=screenheight-50,width=playerwidth,height=playerheight,xvel=20):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.xvel=xvel
        self.pos=(self.x, self.y, self.width, self.height)
        self.hit=0
        self.reward=0
        self.action=0
        objlist.append(self)

    def updatepos(self):
        self.pos = self.x, self.y, self.width, self.height
        
class Wall():
    def __init__(self,x,y,width,height):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.pos=(self.x, self.y, self.width, self.height)
        objlist.append(self)    
        walllist.append(self)    

Wall(-10,0,10,screenheight)
Wall(0,-10,screenwidth,10)
Wall(screenwidth,0,10,screenheight)

class Env():
    def __init__(self):
        self.ball=Ball()
        self.player=Player()
        self.count=0
        self.done=False

    def reset(self):
        del self.ball, self.player
        self.ball, self.player = Ball(), Player()
        self.count=0
        self.done=False
        state = [item*0.01 for item in [self.ball.x, self.ball.y, self.player.x, self.ball.xvel, self.ball.yvel]]
        return state
        
    def playermovement(self):
        self.player.reward=0
        if self.player.action==0: 
            if self.player.x>0:
                self.player.x+=self.player.xvel*-1
                self.player.reward-=0.1 # Penalised for randomly moving
        elif self.player.action==1: 
            pass # No penalty for staying stationary
        elif self.player.action==2: 
            if self.player.x+self.player.width<screenwidth:
                self.player.x+=self.player.xvel
                self.player.reward-=0.1 # Penalised for randomly moving

        if overlap((self.ball.x, self.ball.y, self.ball.width, self.ball.height), 
                   (self.player.x,self.player.y,self.player.width,self.player.height)) > 0:
            if self.ball.yvel>0:
                self.ball.yvel*=-1
                self.player.hit+=1
                self.player.reward+=3 # rewarded for hitting the ball
                self.count+=1

        if self.ball.dead: 
            self.player.reward-=10 # Punished for dying
        self.player.updatepos()


    def runframe(self, action):
        self.done=False
        self.player.action=action
        self.playermovement()
        self.ball.movement()

        state = [item*0.01 for item in [self.ball.x, self.ball.y, self.player.x, self.ball.xvel, self.ball.yvel]]

        if self.ball.dead:
            self.done=True
        if self.count>30:
            self.done=True

        return state, self.player.reward, self.done

    def render(self):
        if win is not None:
            pygame.event.get()
            time.sleep(0.04)
            win.fill((0,0,0))
            pygame.draw.rect(win, (255,255,255), self.ball.pos)
            pygame.draw.rect(win, (255,0,0), self.player.pos)
            pygame.display.update()