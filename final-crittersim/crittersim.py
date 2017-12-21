import numpy as np
import pygame
#import time
import critters as zoo
#from os import listdir
#from os.path import isfile, join
from os.path import join

class World:

    RESOURCE_DIR = "resources"

    def __init__(self, world_width=1024, world_height=1024,
                 num_per_species=50, min_population=True,#lock_population=False,
                 fps=60):
        self.world_width = world_width
        self.world_height = world_height
        self.num_per_species = num_per_species # Starting populations
        self.min_population = min_population
        #self.lock_population = lock_population # Whether or not to force population sizes
        self.fps = fps

        self.pause = True
        self.lock_fps = True
        self.render = True
        self.running = False
        self.controls = [False, False, False, False] # Left, Right, Up, Down

        pygame.init()
        self.screen = pygame.display.set_mode((self.world_width, self.world_height))
        pygame.display.set_caption('Crittersim')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((255,255,255))
        #self.background.fill((0,0,0))
        #pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        #self.resource_names = [f for f in listdir(World.RESOURCE_DIR)
        #                       if isfile(join(World.RESOURCE_DIR, f))]
        self.image_random = self._load_image(join(World.RESOURCE_DIR, "bacteria_forest.png"))
        self.image_ff = self._load_image(join(World.RESOURCE_DIR, "bacteria_blue.png"))
        self.image_player = self._load_image(join(World.RESOURCE_DIR, "bacteria_green.png"))

        self.reset_critters()

    def _load_image(self, image_file):
        try: image = pygame.image.load(image_file)
        except:
            raise RuntimeError, 'Cannot load image: ' + image_file
        image = image.convert_alpha()
        #image.center = (16,32)
        return image

    def reset_critters(self):
        #zoo.CritterPlayer(np.random.randint(self.world_width),
        #                  np.random.randint(self.world_height),
        #                  self.image_player, self, key_list=self.controls)
        for i in range(self.num_per_species):
            zoo.CritterRandom(np.random.randint(self.world_width),
                              np.random.randint(self.world_height),
                              self.image_random, self,
                              size=np.random.random())
        #for i in range(self.num_per_species):
        #    zoo.CritterFF(np.random.randint(self.world_width),
        #                  np.random.randint(self.world_height),
        #                  self.image_random, self)

    def reset_weights(self):
        zoo.Critter.reset_weights()

    def run(self):
        #self.reset_critters()
        self.pause = False
        self.render = True
        self.lock_fps = True
        self.running = True

        while(self.running):
            if self.pause or self.lock_fps: self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.running = False
                    elif event.key == pygame.K_p: self.pause = not self.pause
                    elif event.key == pygame.K_f: self.lock_fps = not self.lock_fps
                    elif event.key == pygame.K_r: self.render = not self.render

                    elif event.key == pygame.K_LEFT: self.controls[0] = True
                    elif event.key == pygame.K_RIGHT: self.controls[1] = True
                    elif event.key == pygame.K_UP: self.controls[2] = True
                    elif event.key == pygame.K_DOWN: self.controls[3] = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT: self.controls[0] = False
                    elif event.key == pygame.K_RIGHT: self.controls[1] = False
                    elif event.key == pygame.K_UP: self.controls[2] = False
                    elif event.key == pygame.K_DOWN: self.controls[3] = False
            
            if self.running and not self.pause:
                zoo.Critter.step()
                zoo.Critter.class_step()
                #print(self.controls)
                #print(zoo.Critter.instances[0].target_angle, self.steps)
                if self.min_population and len(zoo.Critter.instances) < self.num_per_species:
                    zoo.CritterRandom(np.random.randint(self.world_width),
                                      np.random.randint(self.world_height),
                                      self.image_random, self, size=np.random.random()*0.3+0.1)

                if self.render:
                    self.screen.blit(self.background, (0, 0))
                    zoo.Critter.draw()
                    pygame.display.flip()

        self.pause = True

#---------------------------------------------------------------------------#

if __name__ == "__main__":
    w = World()
    w.run()