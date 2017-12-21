import numpy as np
import pygame
from pygame.locals import Rect

class Critter:

    instances = {}
    next_id = 0
    keys = list(instances.keys())
    to_remove = set()
    steps = 0
    
    DRAG = 0.0#0.01
    MAX_VEL = 6 # Taken care of by drag?
    MAX_ACCEL = 1
    MAX_ROT_FRAC = 32
    FRICTION = .01
    FOOD_DIVISOR = 5
    SIZE_DELTA = 0.2
    MAX_SIZE = 1
    BIRTH_SIZE = 0.2
    REPRODUCTION_REWARD = 2
    DEATH_REWARD = -1
    
    #These should be passed in instead:
    IMAGE_SIZE = 32
    HALF_SIZE = IMAGE_SIZE / 2

    @classmethod
    def reset_weights(cls): #<---implicit parameter cls
        pass
        #This one should be specific to each AI class.

    @staticmethod
    def step():
        for k in Critter.keys:
            Critter.instances[k]._step()

        for i in range(len(Critter.to_remove)):
            del Critter.instances[Critter.to_remove.pop()]
        Critter.keys = list(Critter.instances.keys())

        Critter.steps += 1

    # To override:
    @classmethod
    def class_step(cls):
        pass

    @staticmethod
    def draw(): 
        for k in Critter.keys:
            Critter.instances[k]._draw()

    @staticmethod
    def collides(a, b):
        return np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2) < \
            a.bounding_radius + b.bounding_radius

    @staticmethod
    def least_angle_to(angle1, angle2):
        # Expects radians.
        theta = (angle2 - angle1) % (2 * np.pi)
        if theta > np.pi: theta -= 2*np.pi
        #theta -= (theta > np.pi)*(2*np.pi)
        return theta # Expected range: (-pi,pi]

    #def direction_to(self, angle1, angle2):
    #    # Equivalent to np.sign(least_angle_to(angle1, angle2))
    #    if angle1 == angle2: return 0
    #    if (angle2 - angle1) % (2 * np.pi) > np.pi: return -1
    #    else return 1

    @staticmethod
    def rot_center(image, rect, angle):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image,rot_rect


    def __init__(self, x, y, image, world, vel_x=None, vel_y=None, size=BIRTH_SIZE):
        self.my_type = type(self)
        assert self.my_type is not Critter, "Disallowed instantiation of base class Critter."
        self.world = world

        Critter.instances[Critter.next_id] = self        
        self.id = Critter.next_id
        Critter.next_id += 1
        #Critter.keys = list(Critter.instances.keys())

        self.sprite = pygame.sprite.Sprite()
        self.original_image = image
        self.sprite.image = self.original_image
        #self.sprite.rect = self.sprite.image.get_rect()#######
        #self.sprite.rect.center = (Critter.HALF_SIZE, Critter.IMAGE_SIZE)##########
        #self.sprite.center = (Critter.HALF_SIZE, Critter.IMAGE_SIZE)

        self.x = x
        self.y = y
        self.angle = np.random.random() * np.pi * 2
        self.vel_x = 0 if vel_x == None else vel_x
        self.vel_y = 0 if vel_y == None else vel_y
        self.rotate = 0
        self.accelerate = 0
        self.size = size
        self.bounding_radius = size * Critter.HALF_SIZE
        self.drag_multiplier = (1.0-Critter.DRAG) * self.size**(1/8.0)#np.sqrt(self.size)#

        self._update_sprite()

    def _step(self):
        self.rotate, self.accelerate = self._act()
        self.rotate = min(1.0,max(-1.0,self.rotate)) * np.pi / Critter.MAX_ROT_FRAC
        self.accelerate = min(Critter.MAX_ACCEL,max(-Critter.MAX_ACCEL,self.accelerate))

        self.angle += self.rotate
        if self.angle < -np.pi: self.angle += 2*np.pi
        elif self.angle > np.pi: self.angle -= 2*np.pi
        if self.accelerate < 0: fric = Critter.FRICTION - self.accelerate
        else:
            fric = Critter.FRICTION
            self.vel_x += np.cos(self.angle) * self.accelerate
            self.vel_y += np.sin(-self.angle) * self.accelerate

        self.vel_x *= self.drag_multiplier
        self.vel_y *= self.drag_multiplier
        if abs(self.vel_x) > fric: self.vel_x -= fric * np.sign(self.vel_x)
        else: self.vel_x = 0
        if abs(self.vel_y) > fric: self.vel_y -= fric * np.sign(self.vel_y)
        else: self.vel_y = 0

        if self.vel_x > Critter.MAX_VEL: self.vel_x = Critter.MAX_VEL
        elif self.vel_x < -Critter.MAX_VEL: self.vel_x = -Critter.MAX_VEL
        if self.vel_y > Critter.MAX_VEL: self.vel_y = Critter.MAX_VEL
        elif self.vel_y < -Critter.MAX_VEL: self.vel_y = -Critter.MAX_VEL
        self.x += self.vel_x
        self.y += self.vel_y
        self.x = min(self.world.world_width, max(0, self.x))
        self.y = min(self.world.world_height, max(0, self.y))

        for k in Critter.keys:
            if k > self.id:
                if self.collides(self, Critter.instances[k]):
                    if self.size > Critter.instances[k].size + Critter.SIZE_DELTA:
                        self.grow(Critter.instances[k].size)
                        Critter.instances[k].kill()
                    elif self.size + Critter.SIZE_DELTA < Critter.instances[k].size:
                        Critter.instances[k].grow(self.size)
                        self.kill()

        self._update_sprite()

    def _draw(self):
        #if self.id in Critter.keys:
        self.world.screen.blit(self.sprite.image, (self.x, self.y))

    def _update_sprite(self):
        if self.world.render:
            self.sprite.image = pygame.transform.rotozoom(self.original_image, self.angle*180/np.pi - 90, self.size)
            #self.sprite.rect.center = (self.x,self.y)##########
            # NOTE: change the above if start using rect for collisions!
            #self.sprite.image, _ = Critter.rot_center(self.original_image, self.original_image.get_rect(), self.angle*180/np.pi)
            #self.sprite.image = pygame.transform.smoothscale(self.sprite.image, (self.size, self.size))


    #--------------------------------------------------------------------#

    def kill(self, reproduced=False):
        if reproduced: self._train(Critter.REPRODUCTION_REWARD)
        else: self._train(Critter.DEATH_REWARD)
        Critter.to_remove.add(self.id)

    def grow(self, food_size):
        # Food size should range from (0,1)
        self.size += food_size / Critter.FOOD_DIVISOR
        self.bounding_radius = self.size * Critter.HALF_SIZE
        self.drag_multiplier = (1-Critter.DRAG) * np.sqrt(self.size)
        if self.size >= Critter.MAX_SIZE: self.reproduce()
        else: self._train(food_size)

    def reproduce(self):
        for i in range(5):
            self.my_type(self.x + np.random.random()*Critter.IMAGE_SIZE - Critter.HALF_SIZE,
                         self.y + np.random.random()*Critter.IMAGE_SIZE - Critter.HALF_SIZE,
                         self.original_image, self.world,
                         self.vel_x + np.random.random()*2 - 1,
                         self.vel_y + np.random.random()*2 - 1,
                         size=np.random.random()*0.3+0.1)
        self.kill(True)

    #--------------------------------------------------------------------#
    # The following are functions which change according to AI type:

    def _act(self):
        #raise NotImplementedError("_act() not implemented!")
        rotate = 0
        accelerate = 0
        return rotate, accelerate

    def _train(self, reward):
        #loss = -reward
        pass
