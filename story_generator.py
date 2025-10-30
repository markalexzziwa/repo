import random

class StoryGenerator:
    def __init__(self):
        self.story_templates = [
            "Once upon a time, in the lush forests of Uganda, a {species} named {name} soared through the skies. With its {color} feathers gleaming under the sun, it discovered a hidden treasure of {food}. But danger lurked—a sneaky predator approached! Using its clever {behavior}, {name} outsmarted the foe and returned home to its nest, teaching young birds the art of bravery.",
            "In the misty mornings of the savanna, {name} the {species} awoke to the call of adventure. Its {color} plumage blended perfectly with the dawn light as it foraged for {food}. Along the way, it met a fellow traveler and shared tales of {behavior}. Together, they faced a storm, emerging stronger, a symbol of the wild's enduring spirit.",
            "Deep in the heart of Uganda's wilderness, {name} the {species} embarked on a remarkable journey. With {color} accents adorning its wings, it searched far and wide for delicious {food}. Through cunning {behavior}, it navigated challenges and became a legend among the forest creatures.",
            "The {species} known as {name} was no ordinary bird. Its stunning {color} markings made it stand out as it gathered precious {food}. With incredible {behavior}, it taught all who watched about grace and survival in the wild."
        ]
        
        self.behaviors = [
            'swift flight',
            'melodious song',
            'sharp eyesight',
            'agile dance',
            'clever foraging',
            'majestic soaring'
        ]
        
        self.foods = [
            'berries',
            'insects',
            'seeds',
            'nectar',
            'small fruits',
            'fresh buds'
        ]
        
        self.colors = [
            'red', 'blue', 'green', 'yellow',
            'brown', 'black', 'white', 'vibrant',
            'orange', 'purple', 'golden', 'silver'
        ]
    
    def generate_story(self, species, description=None):
        if description:
            colors_in_desc = [c for c in self.colors if c in description.lower()]
            color = random.choice(colors_in_desc) if colors_in_desc else 'vibrant'
        else:
            color = random.choice(self.colors)
        
        behavior = random.choice(self.behaviors)
        food = random.choice(self.foods)
        name = species.split()[0].capitalize() if species else 'Friend'
        
        template = random.choice(self.story_templates)
        story = template.format(
            species=species,
            name=name,
            color=color,
            food=food,
            behavior=behavior
        )
        
        return story
