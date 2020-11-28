import  string
from collections import  Counter
from nltk.corpus import  stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pysentiment2 as ps
from afinn import Afinn
import text2emotion as te
import  pandas as pd
import  random as r

suggestions = {
    'excited': ['https://www.wikihow.com/Manage-Your-Excitement','https://www.youtube.com/watch?v=m3-O7gPsQK0','https://www.youtube.com/watch?v=IPXIgEAGe4U'],
    'focuse' : ['https://www.youtube.com/watch?v=YOJsKatW-Ts','https://www.youtube.com/watch?v=ZfPISsIIKQw'],
    'anxious' : ['https://www.youtube.com/watch?v=ktvTqknDobU','https://www.health.harvard.edu/blog/anxiety-what-it-is-what-to-do-2018060113955'],
    'strong' : ['https://www.youtube.com/watch?v=CVuNpyVXU6E','https://www.youtube.com/watch?v=ZfPISsIIKQw'],
    'respect' : ['Life is too short to waste your time on people who don’t respect, appreciate, and value you','Knowledge give you power but character respect'],
    'confident' : ['https://www.success.com/7-mental-hacks-to-be-more-confident-in-yourself/','https://www.youtube.com/watch?v=Sd-YsMs3OzA&t=110s'],
    'hope' : ['https://theconversation.com/how-hope-can-keep-you-healthier-and-happier-132507','https://www.yourarticlelibrary.com/paragraphs/paragraph-on-hope-the-essence-of-life-by-silki/4228'],
    'relaxed' : ['https://www.youtube.com/watch?v=YOJsKatW-Ts','https://www.youtube.com/watch?v=m3-O7gPsQK0&t=13s'],
    'relieve' : ['https://www.youtube.com/watch?v=ktvTqknDobU','https://www.health.harvard.edu/blog/anxiety-what-it-is-what-to-do-2018060113955'],
    'burdened' : ['https://www.youtube.com/watch?v=cyEdZ23Cp1E','https://www.youtube.com/watch?v=YOJsKatW-Ts','https://www.youtube.com/watch?v=m3-O7gPsQK0&t=13s'],
    'deprived' : ['You can listen to this songs','best of me','Hall of fame','fight back'],
    'ecstatic' : ['You can listen to this songs','best of me','Hall of fame','fight back'],
    'obsessed' :['You can listen to this songs','best of me','Hall of fame','fight back'],
    'independent' : ['You can listen to this songs','best of me','Hall of fame','fight back','Legends never die'],
    'charming' : ['https://www.youtube.com/watch?v=1bumPyvzCyo&t=27s','https://www.youtube.com/watch?v=Zf2RzRgD3t8'],
    'attracted' : ['https://www.youtube.com/watch?v=1bumPyvzCyo&t=27s','https://www.youtube.com/watch?v=Zf2RzRgD3t8'],
    'attached' : ['https://www.youtube.com/watch?v=nLnp0tpZ0ok','https://www.youtube.com/watch?v=UtF6Jej8yb4'],
    'bored' : ['https://www.youtube.com/watch?v=BC3YCpvSFPI','https://www.lifehack.org/articles/communication/10-simple-ways-more-active.html','https://www.wisebread.com/kill-boredom-with-these-34-fun-and-productive-projects'],
    'hurt' : ['https://www.success.com/7-mental-hacks-to-be-more-confident-in-yourself/','https://www.youtube.com/watch?v=BC3YCpvSFPI','https://www.youtube.com/watch?v=wnHW6o8WMas&t=132s'],
    'esteemed' : ['https://www.success.com/7-mental-hacks-to-be-more-confident-in-yourself/','https://www.youtube.com/watch?v=BC3YCpvSFPI','https://www.youtube.com/watch?v=wnHW6o8WMas&'],
    'tensed' : ['https://www.success.com/7-mental-hacks-to-be-more-confident-in-yourself/','https://www.youtube.com/watch?v=BC3YCpvSFPI','https://www.youtube.com/watch?v=wnHW6o8WMas&'],
    'angry' : ["Remember Anger is your biggest Enemy",'https://www.youtube.com/watch?v=PmQ2FJBJBRc',"Anger don't build anything but it only destroy you",'Speak when you are angry and you will make the best speech you will ever regret.'],
    'frustrated' : ['Forget everything and push your-self because no one going to do it for you','https://www.youtube.com/watch?v=wnHW6o8WMas&','https://www.youtube.com/watch?v=Zf2RzRgD3t8','https://www.youtube.com/watch?v=Sd-YsMs3OzA'],
    'belittled' : ['https://www.youtube.com/watch?v=1bumPyvzCyo&t=27s','https://www.youtube.com/watch?v=Zf2RzRgD3t8'],
    'delighted' : ['https://www.youtube.com/watch?v=1bumPyvzCyo&t=27s','https://www.youtube.com/watch?v=ktvTqknDobU','https://www.youtube.com/watch?v=Zf2RzRgD3t8'],
    'safe' : ['https://www.youtube.com/watch?v=jecQcgbyetw','https://www.youtube.com/watch?v=UtF6Jej8yb4','https://www.youtube.com/watch?v=xpVfcZ0ZcFM'],
    'apathetic' : ['https://www.youtube.com/watch?v=jecQcgbyetw','https://www.youtube.com/watch?v=jJPMnTXl63E'],
    'fearless' :['https://www.youtube.com/watch?v=LBr7kECsjcQ','https://www.success.com/7-brave-steps-to-become-fearless/','https://www.youtube.com/watch?v=wnHW6o8WMas&'],
    'surprise' : ['https://www.wikihow.com/Manage-Your-Excitement','https://www.youtube.com/watch?v=m3-O7gPsQK0','https://www.youtube.com/watch?v=IPXIgEAGe4U'],
    'fearful' : ['https://www.youtube.com/watch?v=EP_CDtyV41g','https://www.youtube.com/watch?v=jecQcgbyet'],
    'stressful' : ['https://www.youtube.com/watch?v=eAK14VoY7C0','https://www.youtube.com/watch?v=nKHBIAdBvZ4','https://www.youtube.com/watch?v=9rujCfYXhQc'],
    'depressed' : ['https://www.youtube.com/watch?v=wnHW6o8WMas&','https://www.youtube.com/watch?v=Zf2RzRgD3t8','https://www.youtube.com/watch?v=ptD0T-ZcF2M'],
    'cheated' : ['https://www.youtube.com/watch?v=EP_CDtyV41g','https://www.youtube.com/watch?v=LBr7kECsjcQ','https://www.youtube.com/watch?v=ZAfAud_M_m'],
    'loved' : ['https://www.youtube.com/watch?v=Jekw6EJ-AMo','https://www.youtube.com/watch?v=EN-pgUsBOm4&t=93s','https://www.youtube.com/watch?v=tzBEpdfCRqw'],
    'motivated' :['https://www.youtube.com/watch?v=xOZf52HHKWY','https://www.youtube.com/watch?v=Zf2RzRgD3t8','https://www.youtube.com/watch?v=cNqupwbzTI8'],
    'hated' : ['https://www.youtube.com/watch?v=PmQ2FJBJBRc','https://www.youtube.com/watch?v=zpucCXdOSH8','https://www.youtube.com/watch?v=8_PQJNo2wME'],
    'lost' : ['https://www.youtube.com/watch?v=wnHW6o8WMas&','https://www.youtube.com/watch?v=Sd-YsMs3OzA','https://www.youtube.com/watch?v=hGt_T6bCEMU'],
    'codependent' : ['https://www.youtube.com/watch?v=YOJsKatW-Ts','https://www.youtube.com/watch?v=ZfPISsIIKQw'],
    'embarrassed' : ['https://www.youtube.com/watch?v=_tcZO73dB24','https://blog.iqmatrix.com/overcome-embarrassment'],
    'happy' : ['https://www.youtube.com/watch?v=1bumPyvzCyo&t=27s','https://www.youtube.com/watch?v=ktvTqknDobU','https://www.youtube.com/watch?v=Zf2RzRgD3t8'],
    'sad' : ['https://www.youtube.com/watch?v=eAK14VoY7C0','https://www.youtube.com/watch?v=nKHBIAdBvZ4','https://www.youtube.com/watch?v=9rujCfYXhQc'],
    'admire' : ['Life is too short to waste your time on people who don’t respect, appreciate, and value you','Knowledge give you power but character respect'],
    'demoralized' : ['https://www.success.com/7-mental-hacks-to-be-more-confident-in-yourself/','https://www.youtube.com/watch?v=t8ApMdi24LI','https://www.youtube.com/watch?v=Sd-YsMs3OzA&t=110s'],
}

#STORY

stories = [
    '''The Tiger - Short Story
By Remez Sasson
A teacher and his student were walking from one village to another, when they suddenly heard a roar behind them. Turning their gaze in the direction of the roar, they saw a big tiger following them.
The first thing the student wanted to do was to run away, but since he has been studying and practicing self-discipline, he was able to stop himself from running, and wait to see what his teacher would do.
"What shall we do Master?" Asked the student.


The teacher looked at the student and answered in a calm voice:
"There are several options. We can fill our minds with paralyzing fear so that we cannot move, and let the tiger do with us whatever pleases it. We can let ourselves faint. We can also run away, but then it will run after us.
We can fight it, but physically, the tiger is stronger than us.
We can pray to god to save us. This is another option. We can also send the tiger our love.
There another thing we can do. We can choose to influence the tiger with the power of our mind. However, this requires strong concentration.
We can focus and meditate on the inner power that are within us, and on the fact that we are one with the entire universe, including the tiger, and in this way influence its soul."
"Which option do you choose?" Asked the student. You are the Master. You tell me what to do. We don't have much time."
The master turned his gaze fearlessly toward the tiger, emptied his mind from all thoughts, and entered a deep state of meditation. In his consciousness, he embraced everything in the universe, including the tiger. In this state the consciousness, the teacher became one with consciousness of the tiger.
Meanwhile the student started to shiver with fear, as the tiger was already quite close, ready to make a leap at them. He was amazed at how his teacher could stay so calm and detached in the face of danger.
Meanwhile, the teacher continued to meditate without fear. After a little while the tiger gradually lowered its head and tail, turned around, and went away.
The student was astonished and asked his teacher, "What did you do?"
"Nothing. I just cleared all thoughts from my mind and united myself in spirit with the tiger. We became united in peace on the spiritual level. The tiger sensed the inner calmness, peace, and unity, and felt no threat or need to display violence."
"When the mind is silent and calm, its peace is automatically transmitted to everything and everyone around, influencing them deeply," concluded the teacher.''',
    ''' The Villager and the Happy Man
By Remez Sasson
In a small village in the valley, lived a man who was always happy, kind, and well disposed to everyone he met.
This man always smiled, and had kind and encouraging words to say, whenever it was necessary. Everyone who met him, left feeling better, happier and elated. People knew they could count on him, and regarded him as a great friend.
One of the village dwellers was curious to know what his secret was, and how he could always be so kind and helpful. He wondered, how is it that he held no grudge toward anyone, and always was happy.
Once, upon meeting him in the street he asked him: "Most people are selfish and unsatisfied. They do not smile as often as you do; neither are they as helpful or kind as you are. How do you explain it?"


The man smiled at him and replied, "When you make peace with yourself, you can be in peace with the rest of the world.
If you can recognize the spirit in yourself, you can recognize the spirit in everyone, and then you will find it natural to be kind and well disposed to all."
If your thoughts are under your control, you become strong and firm.
The mind is like a robot programmed to do certain tasks. Habits and thoughts are the tools and programs that control the mind. You need to free yourself from this programming. Then, the inner good and the happiness that reside within you will be revealed."
"But a lot of work is necessary. Good habits have to be developed. The ability to focus and to control the thoughts has to be strengthened. The work is difficult and endless. There are many walls that need to be to climbed. It is not an easy task." Lamented the villager.
"Do not think about the difficulties, otherwise, this is what you will see and experience. Just make your feelings and thoughts quiet, and try to stay in this peace. Just try to be calm, and do not let yourself be carried away by your thoughts."
"Is that all?" Asked the villager.
"Try to watch your thoughts and see how they come and go and stay in the quietness that arises.
The moments of peace will be brief at first, but in time, they will get longer. This peace is also strength, power, kindness, and love.
In time, you will realize that you are one with the Universal Power, and this will lead you to act from a different dimension, different point of view and different consciousness, not from the selfish, small, limited ego."
"I will try to remember your words," said the villager, and continued, "there is another thing that I am curious about. You do not seem to be influenced by the environment. You always seem to be happy. You always have a kind word for everyone, and you are helpful. People treat you well, and never exploit your goodness."
"Some people think that being good and being kind means weakness. The truth is that they mean inner strength, and do not point to weakness.
When you are kind and considerate, you can also be strong.
People sense your inner strength, and therefore, do not impose on you. When you are strong and calm inside, you help people, because you can, and you want to. You act from strength, not from weakness.
Goodness is not a sign of weakness. A good character can manifest together with power and strength.
When you are calm inside, you are also happy. When your mind is quiet, there is no anger, no resentment and no negative thoughts. This leads to happiness and to a feeling of content and joy."
"Thank you very much for your advice and explanations", said the villager, and went away happy and satisfied.''',
    '''The Power of Thoughts - The Yogi and the Disciple
By Remez Sasson
One day, a yogi and his disciple arrived to the big city. They had no money with them, but they needed food and a place to stay. The disciple was sure that they were going to beg for their food, and then sleep in the park at night.
"There is a big park not far from here. We can sleep there at night," said the disciple.
"In the open air?" Asked the yogi.
"Yes", responded the disciple.
The yogi smiled and said: "No, tonight we are going to sleep in a hotel and eat there too".
The disciple was amazed and exclaimed, "We cannot afford that!"
"Come and sit down", said the yogi.
They both sat down on the ground, and the yogi said:
"When you focus your mind intently on any subject, it comes to pass."
The yogi closed his eyes and started to meditate with full concentration. After about ten minutes, he got up and started to walk, with his disciple following him. They walked through several streets and alleys, until they arrived at a hotel.
"Come, let's enter inside." The yogi said to his disciple.
They just set foot at the entrance, when a well-dressed man approached them and said:


"I am the manager of this hotel. You look like traveling swamis, and I believe you have no money. Would you like to work in the kitchen, and in return I'll give you food and a place to stay?"
"Yes, thank you." The yogi responded.
The disciple was perplexed, and he asked the yogi: "Did you use any kind of magic? How did you do that?"
The yogi smiled and said:
"I wanted to show you how the power of thoughts works. When you think with full and strong concentration about something that you want to happen, have full belief that it is going to happen, and do not listen the doubts of your mind, your thought materializes.
"The secret is concentration, visualization with full details, full faith in the power of your thoughts, and projecting mental and emotional energy into the mental scene that you have created in your mind. These are the general prerequisites.
"When your mind is empty from thoughts, and only one single thought is allowed to enter, this one thought gains great power.
"You should be very careful with what you think. A concentrated thought is powerful, and exerts a very strong influence on the environment."
The disciple looked at his teacher and said: "It seems that I have to sharpen my concentration in order to be able to use this power."
"Yes, this is the first step," the yogi replied, "and the next step is to learn to visualize. Then, you will be able to use effectively the power of thoughts.”''',
    ''' The Elephant and the Fly
By Remez Sasson
One day, a disciple and his teacher were walking through the forest. The disciple was disturbed by the fact that his mind was in constant unrest.
He asked his teacher: "Why most people's minds are restless, and only few possess a calm mind? What can one do to still the mind?"
The teacher looked at the disciple, smiled and said, "I will tell you a story."
"On one beautiful day, an elephant was standing by the shade of a tree, eating its leaves. Suddenly, a small fly came buzzing and landed on the elephant's ear. The elephant stayed calm and continued to eat, not heeding the fly."
"The fly flew around the elephant's ear, buzzing noisily, yet the elephant seemed to be unaffected. This bewildered the fly, and it asked, 'Are you deaf?'
"No!" The elephant answered.
"Why aren't you bothered by my buzz?" The fly asked.
"Why are you so restless and noisy? Why can't you stay still just for a few moments?" Asked the elephant, and peacefully continued eating the leaves.
The fly answered, "Everything I see, hear and feel attracts my attention, and all noises and movements around me affect my behavior."
"What is your secret? How can you stay so calm and still?"


The elephant stopped eating and said, "My five senses do not disturb my peace, because they do not rule my attention. I am in control of my mind and my thoughts, and therefore, I can direct my attention where I want, and ignore any disturbances, including your buzz."
"Now that I am eating, I am completely immersed in the act of eating. In this way, I can enjoy my food and chew it better. I am in control of my attention, and therefore, I can stay peaceful."
Upon hearing these words, the disciple's eyes opened wide, and a smile appeared on his face. He looked at his teacher and said:
"I now understand! My mind will always be in constant unrest, if my five senses, and whatever is happening in the world around me, are in control of it. On the other hand, if I am in command of my five senses, able to disregard sense impressions, and able to control my thoughts, my mind will become calm, and I will be able to disregard its restlessness."
"Yes, that's right," answered the teacher,"The mind is restless and goes wherever the attention goes. Control your attention, and you control your mind.”''',
    '''The Mind and the Stormy Ocean
By Remez Sasson
Swamiananda, and his disciple Ranga, were strolling on the beach by the ocean. It was a cold day and the wind was blowing strongly over the ocean, raising very high waves.
After walking for some time, Swamiananda stopped, looked at his disciple and asked:
"What does the choppy ocean remind you of?"
"It reminds me of my mind and my rushing and restless thoughts." Answered Ranga.
"Yes, the stormy ocean is like the mind, and the waves are the thoughts." Swamiananda explained. "The mind is neutral like the water. It is neither good, nor bad. The wind is creating the waves, as desires and fears produce thoughts."
"I wouldn't want to be on a boat in the middle of the ocean, in a storm like this." Said Ranga.
"You are there all the time within this storm." Responded Swamiananda and continued, "Most people are on a rudderless boat in the middle of a choppy ocean, even if they do not realize it. The mind of most people is very restless. Thoughts of all kinds come and go incessantly, agitating the mind like the ocean's waves."
"Yes," Ranga interrupted him, "You don't need to tell me this, I know that. This is the reason I want to learn from you. I want to calm down the waves of my mind."
Swamiananda looked at Ranga for a while, smiled, and said:
"You don't calm the ocean by holding the water and not letting it move. What is necessary is to stop the wind. Your thoughts, desires and fears are like the wind, and you need to calm them down, and not let them rule your life. You learn to control them by controlling your attention and focus, and then the ocean of your mind would become calm."
"And how do I do that?"
"Suppose it is possible for the ocean to disregard the wind, what would happen then?" asked Swamiananda.
"The waves would cease. However, no one can stop the wind."
Swamiananda looked at Ranga with a mysterious smile and said:

"You can calm down the winds in your mind, which make the ocean of your mind restless. The winds are your thoughts and the ocean is your mind."
"Yes master", said Ranga, "this is what I am trying to do. If I can succeed to calm the windows of my mind, would I also be able to bring more peace and calmness into the world around me?"
"First, learn to calm down the wind and the ocean in your mind," Swamiananda explained. "When you can control your mind and make it peaceful, you will have more control of your life. However, don't focus on changing the outside world. Focus on changing and calming your inner world."
"After you are able to control your inner world, you would be able to control the world around you."
"And how do I do that?" Ranga asked.
"Learn to focus your mind, develop willpower and self-discipline, and learn to meditate." Swamiananda answered.''',
    ''' Two Cats Facing Each Other on the Path
By Remez Sasson
Two cats were walking on a narrow path toward each other. When they came near one another, no one was willing to let the other pass first. They just stood there, screaming at each other.
"You let me pass first," said one cat.
"No! I was first here," said the other cat.
"No, I must be first, because I am bigger."
"No, I must be first, because I am more beautiful."
"No, I am wiser than you, and you must, therefore, respect me."


"I am stronger."
"I have many cats that will hurt you if you don't let me pass."
After a while, the screams turned into a fight. The cats started fighting, scratching, and biting each other.
A little while later, a wiser cat arrived to the scene. He looked at them and started to laugh.
The two cats stopped fighting, at looked at him amazed.
"Why are you laughing?" The cats asked him.
"I am laughing at you and at your behavior. You are wasting your time and are hurting each other, just because you won't let the other one pass. The path is wide enough for each cat to pass to the other side."
"Why are you fighting? Don't you have anything better to do?"
"It is a matter of honor and power." The two cats said.
The wise cat was amused and said, "Do you need to prove you are stronger? Who cares about this?"
"Someone, who is really strong and self-confident, doesn't feel the need to show this to others. He or she feels good about himself, and others feel his strength, and respect him, with love, not fear."
"There is life, there is good food, there are wonderful things to enjoy and to do, and you are standing, here, facing each other, yelling, scratching and fighting. Is this a practical and rational act? Open your eyes and grow up!"
"Is it really important who passes first to the other side of the path? Is it worthwhile to have all these scratches and bites? You are wasting your time, energy and health on nonsense."
"Look around, and see all the animals around you laughing at your irrational behavior."
The two cats were bewildered and didn't know what to say. The words of the wise cat made sense, but their subconscious, programmed behavior and habits were too strong. It was not easy for them to resist them.
Did the two cats stopped fighting and each one went on their way? I leave that for you to think about.''',
    '''The Swamis and the Mysterious Light
By Remez Sasson

A long time ago, there were two swamis (monks), who lived in two neighboring caves.
The Swamis spent most of their time in deep meditation, except the time they ate or were visited by devotees.
The people who came to visit them revered the two swamis, and loved listening to their teachings and advice. They always felt peaceful and happy when near them, a feeling that continued, even after they went away.
One cave was dark, as caves usually are, but in the other one, sometimes, there was a peculiar golden light illuminating the cave. It was not strong, but enough to be noticed and to illuminate the cave.
The phenomenon of the light bewildered the visitors, but they could not come to an agreement about its causes. Both swamis were rather silent most of the time, and did not want to discuss the phenomenon of the light.


Being in the company of the swamis aroused calmness and peace in the visitors. Their minds slowed their nonstop chatter, and they experienced a pleasant inner peace and happiness.
Though the visitors admired both swamis, they believed that the one living in the illuminated cave possessed supernatural powers and was more advanced. He certainly appeared to them as a mysterious person.
One day, a great and well-known sage arrived to a nearby village. One of the villagers came to him and said:
"Great master, we have a question to ask. There is a mystery which you might solve for us."
"I will be glad to help you, if I can." Answered the sage.
"There are two swamis living here on the hill...", the villager started to recount.
"Yes, I know," the sage interrupted him, "and you want to know about the light in the cave."
"Yes, that is true," the villager responded, "It is something that has been a riddle for us. Can you please tell us also, if the swami in the lighted cave is more advanced, and if he really possess supernatural powers?"
"Pay attention to your inner self and not to the external phenomena."The sane said, and continued, "The outside world always changes, but the inner self does not. When in the presence of a teacher, listen to what he says and be aware of the influence of his words on you. Watch yourself, and see whether under his influence you become calmer and more peaceful, and whether your thoughts, at least for a while, slow down their mad race."

"Yes, I'll do so," said the villager, "but please enlightens us on the mysterious light."
The sage sat down, and started to explain:
"Sometimes, when one works intensively on the spiritual path, and concentrates and meditates a lot, various phenomena may occur around him, such as lights, sounds or visions. This is not supernatural. The mind has a creative power, and when concentrated, can produce various phenomena even unintentionally."
"This does not mean that one is more advanced than the other. Not all minds produce these things. Some do, and some don't."
"Some of the people who produce these lights may be aware of the light, and others might not. This depends on their psychic sensitivity. So it is also with the people who watch them. Not all see this light. In any case, it has nothing to do with whether one swami is more advanced or less advanced than the other one."
"Thank you great master, you have solved us this great mystery." Exclaimed the devotees of the swamis, who were standing by, deeply relieved and happy to understand the mystery that has been bewildering them for a long time.
''',
    '''The Elephant Rope (Belief)

A gentleman was walking through an elephant camp, and he spotted that the elephants weren’t being kept in cages or held by the use of chains.
All that was holding them back from escaping the camp, was a small piece of rope tied to one of their legs.
As the man gazed upon the elephants, he was completely confused as to why the elephants didn’t just use their strength to break the rope and escape the camp. They could easily have done so, but instead, they didn’t try to at all.
Curious and wanting to know the answer, he asked a trainer nearby why the elephants were just standing there and never tried to escape.
The trainer replied;
 “when they are very young and much smaller we use the same size rope to tie them and, at that age, it’s enough to hold them. As they grow up, they are conditioned to believe they cannot break away. They believe the rope can still hold them, so they never try to break free.”
The only reason that the elephants weren’t breaking free and escaping from the camp was that over time they adopted the belief that it just wasn’t possible.
Moral of the story:
No matter how much the world tries to hold you back, always continue with the belief that what you want to achieve is possible. Believing you can become successful is the most important step in actually achieving it.
''',
    '''Thinking Out of the Box (Creative Thinking)
In a small Italian town, hundreds of years ago, a small business owner owed a large sum of money to a loan-shark. The loan-shark was a very old, unattractive looking guy that just so happened to fancy the business owner’s daughter.
He decided to offer the businessman a deal that would completely wipe out the debt he owed him. However, the catch was that we would only wipe out the debt if he could marry the businessman’s daughter.
Needless to say, this proposal was met with a look of disgust.
The loan-shark said that he would place two pebbles into a bag, one white and one black.
The daughter would then have to reach into the bag and pick out a pebble. If it was black, the debt would be wiped, but the loan-shark would then marry her. If it was white, the debt would also be wiped, but the daughter wouldn’t have to marry the loan-shark.
Standing on a pebble-strewn path in the businessman’s garden, the loan-shark bent over and picked up two pebbles.
Whilst he was picking them up, the daughter noticed that he’d picked up two black pebbles and placed them both into the bag.
He then asked the daughter to reach into the bag and pick one.
The daughter naturally had three choices as to what she could have done:
1.Refuse to pick a pebble from the bag.
2.Take both pebbles out of the bag and expose the loan-shark for cheating.
3.Pick a pebble from the bag fully well knowing it was black and sacrifice herself for her father’s freedom.
She drew out a pebble from the bag, and before looking at it ‘accidentally’ dropped it into the midst of the other pebbles. She said to the loan-shark; 
“Oh, how clumsy of me. Never mind, if you look into the bag for the one that is left, you will be able to tell which pebble I picked.”
 
The pebble left in the bag is obviously black, and seeing as the loan-shark didn’t want to be exposed, he had to play along as if the pebble the daughter dropped was white, and clear her father’s debt.
 
Moral of the story:
It’s always possible to overcome a tough situation throughout of the box thinking, and not give in to the only options you think you have to pick from.
 ''',
    '''The Group of Frogs (Encouragement)
As a group of frogs was traveling through the woods, two of them fell into a deep pit. When the other frogs crowded around the pit and saw how deep it was, they told the two frogs that there was no hope left for them.
However, the two frogs decided to ignore what the others were saying and they proceeded to try and jump out of the pit. 
Despite their efforts, the group of frogs at the top of the pit were still saying that they should just give up. That they would never make it out.
Eventually, one of the frogs took heed to what the others were saying and he gave up, falling down to his death. The other frog continued to jump as hard as he could. Again, the crowd of frogs yelled at him to stop the pain and just die.
He jumped even harder and finally made it out. When he got out, the other frogs said, “Did you not hear us?”
The frog explained to them that he was deaf. He thought they were encouraging him the entire time.
 
Moral of the story:
People’s words can have a big effect on other’s lives. Think about what you say before it comes out of your mouth. It might just be the difference between life and death.''',
    '''A Pound of Butter (Honesty)
There was a farmer who sold a pound of butter to a baker. One day the baker decided to weigh the butter to see if he was getting the right amount, which he wasn’t. Angry about this, he took the farmer to court.
The judge asked the farmer if he was using any measure to weight the butter. The farmer replied, “Honor, I am primitive. I don’t have a proper measure, but I do have a scale.”
The judge asked, “Then how do you weigh the butter?”
The farmer replied;
“Your Honor, long before the baker started buying butter from me, I have been buying a pound loaf of bread from him. Every day when the baker brings the bread, I put it on the scale and give him the same weight in butter. If anyone is to be blamed, it is the baker.”
Moral of the story:
In life, you get what you give. Don’t try and cheat others.
 ''',
    ''' The Obstacle In Our Path (Opportunity)
In ancient times, a King had a boulder placed on a roadway. He then hid himself and watched to see if anyone would move the boulder out of the way. Some of the king’s wealthiest merchants and courtiers came by and simply walked around it.
Many people loudly blamed the King for not keeping the roads clear, but none of them did anything about getting the stone out of the way.
A peasant then came along carrying a load of vegetables. Upon approaching the boulder, the peasant laid down his burden and tried to push the stone out of the road. After much pushing and straining, he finally succeeded.
After the peasant went back to pick up his vegetables, he noticed a purse lying in the road where the boulder had been.
The purse contained many gold coins and a note from the King explaining that the gold was for the person who removed the boulder from the roadway.
Moral of the story:
Every obstacle we come across in life gives us an opportunity to improve our circumstances, and whilst the lazy complain, the others are creating opportunities through their kind hearts, generosity, and willingness to get things done. 
''',
    ''' The Butterfly (Struggles)
A man found a cocoon of a butterfly.
One day a small opening appeared. He sat and watched the butterfly for several hours as it struggled to force its body through that little hole.
Until it suddenly stopped making any progress and looked like it was stuck.
So the man decided to help the butterfly. He took a pair of scissors and snipped off the remaining bit of the cocoon. The butterfly then emerged easily, although it had a swollen body and small, shriveled wings.
The man didn’t think anything of it and sat there waiting for the wings to enlarge to support the butterfly. But that didn’t happen. The butterfly spent the rest of its life unable to fly, crawling around with tiny wings and a swollen body.
Despite the kind heart of the man, he didn’t understand that the restricting cocoon and the struggle needed by the butterfly to get itself through the small opening; were God’s way of forcing fluid from the body of the butterfly into its wings. To prepare itself for flying once it was out of the cocoon.
 
Moral of the story:
Our struggles in life develop our strengths. Without struggles, we never grow and never get stronger, so it’s important for us to tackle challenges on our own, and not be relying on help from others.
 ''',
    ''' Control Your Temper (Anger)
There once was a little boy who had a very bad temper. His father decided to hand him a bag of nails and said that every time the boy lost his temper, he had to hammer a nail into the fence.
On the first day, the boy hammered 37 nails into that fence.
The boy gradually began to control his temper over the next few weeks, and the number of nails he was hammering into the fence slowly decreased.
He discovered it was easier to control his temper than to hammer those nails into the fence.
Finally, the day came when the boy didn’t lose his temper at all. He told his father the news and the father suggested that the boy should now pull out a nail every day he kept his temper under control.
The days passed and the young boy was finally able to tell his father that all the nails were gone. The father took his son by the hand and led him to the fence.
“you have done well, my son, but look at the holes in the fence. The fence will never be the same. When you say things in anger, they leave a scar just like this one. You can put a knife in a man and draw it out. It won’t matter how many times you say I’m sorry, the wound is still there.”
 
Moral of the story:
Control your anger, and don’t say things to people in the heat of the moment, that you may later regret. Some things in life, you are unable to take back.''',
    '''Puppies for Sale (Understanding)
A shop owner placed a sign above his door that said: “Puppies For Sale.”
Signs like this always have a way of attracting young children, and to no surprise, a boy saw the sign and approached the owner; 
 “How much are you going to sell the puppies for?” he asked. 
The store owner replied, “Anywhere from $30 to $50.”
The little boy pulled out some change from his pocket. “I have $2.37,” he said. “Can I please look at them?”
The shop owner smiled and whistled. Out of the kennel came Lady, who ran down the aisle of his shop followed by five teeny, tiny balls of fur.
One puppy was lagging considerably behind. Immediately the little boy singled out the lagging, limping puppy and said, “What’s wrong with that little dog?”
The shop owner explained that the veterinarian had examined the little puppy and had discovered it didn’t have a hip socket. It would always limp. It would always be lame.
The little boy became excited. “That is the puppy that I want to buy.”
The shop owner said, “No, you don’t want to buy that little dog. If you really want him, I’ll just give him to you.”
The little boy got quite upset. He looked straight into the store owner’s eyes, pointing his finger, and said;
“I don’t want you to give him to me. That little dog is worth every bit as much as all the other dogs and I’ll pay full price. In fact, I’ll give you $2.37 now, and 50 cents a month until I have him paid for.”
The shop owner countered, “You really don’t want to buy this little dog. He is never going to be able to run and jump and play with you like the other puppies.”
To his surprise, the little boy reached down and rolled up his pant leg to reveal a badly twisted, crippled left leg supported by a big metal brace. He looked up at the shop owner and softly replied, “Well, I don’t run so well myself, and the little puppy will need someone who understands!”''',
    ''' Box Full of Kisses (Love)
Some time ago, a man punished his 3-year-old daughter for wasting a roll of gold wrapping paper. Money was tight and he became infuriated when the child tried to decorate a box to put under the Christmas tree.
Nevertheless, the little girl brought the gift to her father the next morning and said, “This is for you, Daddy.”
The man became embarrassed by his overreaction earlier, but his rage continue when he saw that the box was empty. He yelled at her; “Don’t you know, when you give someone a present, there is supposed to be something inside?”
The little girl looked up at him with tears in her eyes and cried;
 “Oh, Daddy, it’s not empty at all. I blew kisses into the box. They’re all for you, Daddy.”
The father was crushed. He put his arms around his little girl, and he begged for her forgiveness.
Only a short time later, an accident took the life of the child.
Her father kept the gold box by his bed for many years and, whenever he was discouraged, he would take out an imaginary kiss and remember the love of the child who had put it there.
 
Moral of the story:
Love is the most precious gift in the world.''',
    ''' Everyone Has a Story in Life
A 24 year old boy seeing out from the train’s window shouted…
“Dad, look the trees are going behind!”
Dad smiled and a young couple sitting nearby, looked at the 24 year old’s childish behavior with pity, suddenly he again exclaimed…
“Dad, look the clouds are running with us!”
The couple couldn’t resist and said to the old man…
“Why don’t you take your son to a good doctor?” The old man smiled and said…“I did and we are just coming from the hospital, my son was blind from birth, he just got his eyes today.”
Every single person on the planet has a story. Don’t judge people before you truly know them. The truth might surprise you.''',
    '''Shake off Your Problems
A man’s favorite donkey falls into a deep precipice. He can’t pull it out no matter how hard he tries. He therefore decides to bury it alive.
Soil is poured onto the donkey from above. The donkey feels the load, shakes it off, and steps on it. More soil is poured.
It shakes it off and steps up. The more the load was poured, the higher it rose. By noon, the donkey was grazing in green pastures.
After much shaking off (of problems) And stepping up (learning from them), One will graze in GREEN PASTURES.''',
    '''Potatoes, Eggs, and Coffee Beans
Once upon a time a daughter complained to her father that her life was miserable and that she didn’t know how she was going to make it. She was tired of fighting and struggling all the time. It seemed just as one problem was solved, another one soon followed.
Her father, a chef, took her to the kitchen. He filled three pots with water and placed each on a high fire. Once the three pots began to boil, he placed potatoes in one pot, eggs in the second pot, and ground coffee beans in the third pot.
He then let them sit and boil, without saying a word to his daughter. The daughter, moaned and impatiently waited, wondering what he was doing.
After twenty minutes he turned off the burners. He took the potatoes out of the pot and placed them in a bowl. He pulled the eggs out and placed them in a bowl.
He then ladled the coffee out and placed it in a cup. Turning to her he asked. “Daughter, what do you see?”
“Potatoes, eggs, and coffee,” she hastily replied.
“Look closer,” he said, “and touch the potatoes.” She did and noted that they were soft. He then asked her to take an egg and break it. After pulling off the shell, she observed the hard-boiled egg. Finally, he asked her to sip the coffee. Its rich aroma brought a smile to her face.
“Father, what does this mean?” she asked.
He then explained that the potatoes, the eggs and coffee beans had each faced the same adversity– the boiling water.
However, each one reacted differently.
The potato went in strong, hard, and unrelenting, but in boiling water, it became soft and weak.
The egg was fragile, with the thin outer shell protecting its liquid interior until it was put in the boiling water. Then the inside of the egg became hard.
However, the ground coffee beans were unique. After they were exposed to the boiling water, they changed the water and created something new.
“Which are you,” he asked his daughter. “When adversity knocks on your door, how do you respond? Are you a potato, an egg, or a coffee bean? “
Moral:In life, things happen around us, things happen to us, but the only thing that truly matters is what happens within us.''',
    '''A Dish of Ice Cream
In the days when an ice cream sundae cost much less, a 10 year old boy entered a hotel coffee shop and sat at a table. A waitress put a glass of water in front of him.
“How much is an ice cream sundae?”
“50 cents,” replied the waitress.
The little boy pulled his hand out of his pocket and studied a number of coins in it.
“How much is a dish of plain ice cream?” he inquired. Some people were now waiting for a table and the waitress was a bit impatient.
“35 cents,” she said brusquely.
The little boy again counted the coins. “I’ll have the plain ice cream,” he said.
The waitress brought the ice cream, put the bill on the table and walked away. The boy finished the ice cream, paid the cashier and departed.
When the waitress came back, she began wiping down the table and then swallowed hard at what she saw.
There, placed neatly beside the empty dish, were 15 cents – her tip.''',
    ''' Colonel Sanders | Kentucky Fried Chicken
Once, there was an older man, who was broke, living in a tiny house and owned a beat up car. He was living off of $99 social security checks. At 65 years of age, he decide things had to change. So he thought about what he had to offer. His friends raved about his chicken recipe. He decided that this was his best shot at making a change.
He left Kentucky and traveled to different states to try to sell his recipe. He told restaurant owners that he had a mouthwatering chicken recipe. He offered the recipe to them for free, just asking for a small percentage on the items sold. Sounds like a good deal, right?
Unfortunately, not to most of the restaurants. He heard NO over 1000 times. Even after all of those rejections, he didn’t give up. He believed his chicken recipe was something special. He got rejected 1009 times before he heard his first yes.
With that one success Colonel Hartland Sanders changed the way Americans eat chicken. Kentucky Fried Chicken, popularly known as KFC, was born.
Remember, never give up and always believe in yourself in spite of rejection.''',
    '''The Obstacle in our Path
There once was a very wealthy and curious king. This king had a huge boulder placed in the middle of a road. Then he hid nearby to see if anyone would try to remove the gigantic rock from the road.
The first people to pass by were some of the king’s wealthiest merchants and courtiers. Rather than moving it, they simply walked around it. A few loudly blamed the King for not maintaining the roads. Not one of them tried to move the boulder.
Finally, a peasant came along. His arms were full of vegetables. When he got near the boulder, rather than simply walking around it as the others had, the peasant put down his load and tried to move the stone to the side of the road. It took a lot of effort but he finally succeeded.
The peasant gathered up his load and was ready to go on his way when he say a purse lying in the road where the boulder had been. The peasant opened the purse. The purse was stuffed full of gold coins and a note from the king. The king’s note said the purse’s gold was a reward for moving the boulder from the road.
The king showed the peasant what many of us never understand: every obstacle presents an opportunity to improve our condition.
''',
    '''Value
A popular speaker started off a seminar by holding up a $20 bill. A crowd of 200 had gathered to hear him speak. He asked, “Who would like this $20 bill?”
200 hands went up.
He said, “I am going to give this $20 to one of you but first, let me do this.” He crumpled the bill up.
He then asked, “Who still wants it?”
All 200 hands were still raised.
“Well,” he replied, “What if I do this?” Then he dropped the bill on the ground and stomped on it with his shoes.
He picked it up, and showed it to the crowd. The bill was all crumpled and dirty.
“Now who still wants it?”
All the hands still went up.
“My friends, I have just showed you a very important lesson. No matter what I did to the money, you still wanted it because it did not decrease in value. It was still worth $20. Many times in our lives, life crumples us and grinds us into the dirt. We make bad decisions or deal with poor circumstances. We feel worthless. But no matter what has happened or what will happen, you will never lose your value. You are special – Don’t ever forget it!
''',
'''A Very Special Bank Account
Imagine you had a bank account that deposited $86,400 each morning. The account carries over no balance from day to day, allows you to keep no cash balance, and every evening cancels whatever part of the amount you had failed to use during the day. What would you do? Draw out every dollar each day!
We all have such a bank. Its name is Time. Every morning, it credits you with 86,400 seconds. Every night it writes off, as lost, whatever time you have failed to use wisely. It carries over no balance from day to day. It allows no overdraft so you can’t borrow against yourself or use more time than you have. Each day, the account starts fresh. Each night, it destroys an unused time. If you fail to use the day’s deposits, it’s your loss and you can’t appeal to get it back.
There is never any borrowing time. You can’t take a loan out on your time or against someone else’s. The time you have is the time you have and that is that. Time management is yours to decide how you spend the time, just as with money you decide how you spend the money. It is never the case of us not having enough time to do things, but the case of whether we want to do them and where they fall in our priorities.
Motivational Weight Loss Stories
Losing weight feels like a constant uphill battle. Shedding stubborn pounds and getting healthy can be a lot of hard work. During a plateau or at the beginning of the a weight loss journey, it’s easy to be discouraged. Find ways to motivate yourself, like reading these inspirational weight loss stories. Know if they can do it, so can you!
''',
'''Unstoppable
Extreme Makeover features a celebrity trainer helping very overweight individuals reach their weight loss goals. Sometimes, their attitudes aren’t great, but other times, the people on the show are truly amazing, like Sara. Sara is a little person, standing at only 4’5″. She was a nutrition speaker on local television shows at the start of her journey, but ashamed of herself. Not only had she spent her life dealing with her short stature, but she had suffered greatly at the hands of her sister. She turned to eating and by the time she was 37 years old, weighed over 200 pounds.
When she began her time on Extreme Makeover, her first challenge was to climb the stairs of an amphitheater holding an 80 pound weight. The stairs came up past her knees. But she didn’t complain once. She kept going. Slowly, all the people in the theater started to watch her. By the time she reached the last step, the crowd cheered for her.
Her trainer gave her the goal to run a half marathon 6 months after starting her diet and exercise program. Sara said no. She wouldn’t run the half. Instead she would run a full marathon. Her trainer advised against it because it would be extra hard on her body. She’d have to take many extra strides due to her short stature. Sara didn’t care. She ran the whole marathon.
She succeeded in loosing more than half her body weight and becoming a runner, like she had always dreamed.
''',
'''Winning the Battle
Adrienne Brown shared her weight loss journey with Good Housekeeping. Adrienne loved to eat and was a bit food obsessed. As an adult, she owned two refrigerators stocked with food. She was already overweight at 180 pounds when her weight shot up as she battled breast cancer.
Adrienne got serious about her health during her battle with cancer. Inspired by Jennifer Garner in Alias and determined to be healthier, Adrienne lost 90 pounds in a year by eliminating processed foods and exercising. She made it manageable by breaking her goal into 10 pound increments and keeping a positive attitude.
''',
'''The Weight Was Wrong
Drew Carey has spent much of his career in the spotlight. Fans of the comedic actor remember him as the overweight star of the Drew Carey Show. He shocked his following by appearing on his new job as the host of the The Price is Right, a full 80 pounds lighter. There was no magic trick to his weight loss. Carey lost weight the old fashioned way, by counting calories and logging 45 minute cardio sessions on the treadmill.
Remember, you don’t have to try the newest fad (The 3 Day Military Diet). Find what works for you and just stick with it. Little by little you will reach your goals.
''',
'''The Dean Schooled Them
One night four college kids stayed out late, partying and having a good time. They paid no mind to the test they had scheduled for the next day and didn’t study. In the morning, they hatched a plan to get out of taking their test. They covered themselves with grease and dirt and went to the Dean’s office. Once there, they said they had been to a wedding the previous night and on the way back they got a flat tire and had to push the car back to campus.
The Dean listened to their tale of woe and thought. He offered them a retest three days later. They thanked him and accepted his offer.hat time.
When the test day arrived, they went to the Dean. The Dean put them all in separate rooms for the test. They were fine with this since they had all studied hard. Then they saw the test. It had 2 questions.
1) Your Name __________ (1 Points)
2) Which tire burst? __________ (99 Points)
Options – (a) Front Left (b) Front Right (c) Back Left (d) Back Right
The lesson: always be responsible and make wise decisions.''',
'''The Right Place
A mother and a baby camel were lying around under a tree.
Then the baby camel asked, “Why do camels have humps?”
The mother camel considered this and said, “We are desert animals so we have the humps to store water so we can survive with very little water.”
The baby camel thought for a moment then said, “Ok…why are our legs long and our feet rounded?”
The mama replied, “They are meant for walking in the desert.”
The baby paused. After a beat, the camel asked, “Why are our eyelashes long? Sometimes they get in my way.”
The mama responded, “Those long thick eyelashes protect your eyes from the desert sand when it blows in the wind.
The baby thought and thought. Then he said, “I see. So the hump is to store water when we are in the desert, the legs are for walking through the desert and these eye lashes protect my eyes from the desert then why in the Zoo?”
The Lesson: Skills and abilities are only useful if you are in the right place at the right time. Otherwise they go to waste.''',
'''On God’s Time
A man walked to the top of a hill to talk to God.
The man asked, “God, what’s a million years to you?” and God said, “A minute.”
Then the man asked, “Well, what’s a million dollars to you?” and God said, “A penny.”
Then the man asked, “God…..can I have a penny?” and God said, “Sure…..in a minute.”''',
'''Puppy Love
The Story:
A pet shop owner got a new litter of puppies and was ready to sell them to their “forever” families. A young girl walked by the shop and noticed a sign saying, “Puppies for Sale” and of course was very eager to go inside.
She asked the owner, “How much do the puppies cost?” The owner replied, “They are all around $50.”
The girl emptied her pocket change and told the store owner that she only had about $2, but she still wanted to look at them.
The shop owner whistled for the dogs, who came running down the hall of his shop. Five tiny furballs, followed by one, limping behind the rest. The girl immediately singled out the lagging puppy and asked the store owner what was wrong with him.
The owner explained that the puppy was born with a deformity– he was missing a hip socket. He would walk with a limp for the rest of his life.
The girl got excited, saying, “I want that puppy!”
The owner replied, “You don’t want to buy that puppy. If you really want him, you can have him for free.”
The girl became upset. She looked at the owner and said, “I don’t want to have him for free. That puppy is worth just as much as the others. I’ll give you the change I have now and a dollar a month until I have paid for the puppy entirely.”
The owner continued, “This dog is never going to be able to run and play like all of the other dogs, I think you’re going to regret this decision.”
To his surprise, the girl reached down and rolled up her pant leg to reveal a crippled leg that was supported by a large metal brace. She looked up at the owner and softly replied, ‘Well, I’m not much of a runner, and this puppy needs someone who understands.”
The Moral:
Don’t make assumptions about other people’s wants, needs, or abilities. Every one of us has our own weaknesses, whether it’s physical or mental. The trick is to not allow your weaknesses to slow you down, and instead, find others in the world who can support you. Find and surround yourself with people who challenge you to reach your potential.
''',
'''Cleaning Turtles
The Story:
There was once a man who walked his dog every Sunday morning around a lake near his house. Week after week, he saw the same elderly woman sitting at the edge of the water with a small metal cage next to her.
The man’s curiosity finally got the best of him and he approached the woman one day. He noticed that the cage was actually a small trap and she had three small turtles in it. In her lap, there was a fourth turtle that she was carefully wiping down with a sponge.
The man greeted her and said, “If you don’t mind my asking, what do you do with these turtles every week?”
She smiled and explained to him that she was cleaning their shells because any algae or scum that builds up on a turtle’s shell reduces its ability to absorb heat and slows down their swimming. It can also corrode their shell and weaken it over time.
The man was impressed as the woman continued, “I do this every Sunday morning to help the turtles.”
“But don’t most turtles live their entire lives with algae on their shells?” the man asked.
The woman agreed that was true.
He replied, “Well then, you’re kind to do this, but are you really making a difference if most turtles don’t have people around to clean their shells?”
The woman laughed as she looked down at the small turtle on her lap. “Young man, if this little turtle could talk, he would say I’m making all the difference in the world.'”
The Moral:
“To the world you may be one person; but to one person you may be the world.” — Dr. Seuss
Just because you may not be able to change the world or help everyone, you can still make a huge difference in one person’s life by offering them any help that you can. Don’t choose to not do anything because you can’t do everything.
The actions of one person can make a world of difference to someone else. When you see someone in need, you may never know how much of a difference your help can make in their life.
''',
'''The Chef’s Daughter
The Story:
Once there was a girl who was complaining to her dad that her life was so hard and that she didn’t know how she would get through all of her struggles. She was tired, and she felt like as soon as one problem was solved, another would arise.
Being a chef, the girl’s father took her into his kitchen. He boiled three pots of water that were equal in size. He placed potatoes in one pot, eggs in another, and ground coffee beans in the final pot.
He let the pots sit and boil for a while, not saying anything to his daughter.
He turned the burners off after twenty minutes and removed the potatoes from the pot and put them in a bowl. He did the same with the boiled eggs. He then used a ladle to scoop out the boiled coffee and poured it in a mug. He asked his daughter, “What do you see?”
She responded, “Potatoes, eggs, and coffee.”
Her father told her to take a closer look and touch the potatoes. After doing so, she noticed they were soft. Her father then told her to break open an egg. She acknowledged the hard-boiled egg. Finally, he told her to take a sip of the coffee. It was rich and delicious.
After asking her father what all of this meant, he explained that each of the three food items had just undergone the exact same hardship–twenty minutes inside of boiling water.
However, each item had a different reaction.
The potato went into the water as a strong, hard item, but after being boiled, it turned soft and weak.
The egg was fragile when it entered the water, with a thin outer shell protecting a liquid interior. However, after it was left to boil, the inside of the egg became firm and strong.
Finally, the ground coffee beans were different. Upon being exposed to boiling water, they changed the water to create something new altogether.
He then asked his daughter, “Which are you? When you face adversity, do you respond by becoming soft and weak? Do you build strength? Or do you change the situation?”
The Moral:
Life is full of ups and downs, wins and losses, and big shifts in momentum, and adversity is a big part of this experience. And while many of us would rather not face adversity, it doesn’t have to always be a negative thing. In fact, handling adversity can be a positive experience that can lead to personal development.
You choose how you respond to adversity, whether you let it break you down or you stand up in the face of it and learn from it. In many instances, facing adversity gives you a chance to learn important lessons that can help you grow as a person.
When facing adversity, it’s important to recognize your freedom to choose how you respond. You can respond in a way that ultimately limits you, or you can choose to have a more productive response that could potentially open windows of opportunity that we didn’t know existed.
''',
'''Don’t Hold Back
The Story:
There was once a company whose CEO was very strict and often disciplined the workers for their mistakes or perceived lack of progress. One day, as the employees came into work, they saw a sign on the door that read, “Yesterday, the person who has been holding you back from succeeding in this company passed away. Please gather for a funeral service in the assembly room.”
While the employees were saddened for the family of their CEO, they were also intrigued at the prospect of being able to now move up within the company and become more successful.”
Upon entering the assembly room, many employees were surprised to see the CEO was, in fact, present. They wondered among themselves, “If it wasn’t him who was holding us back from being successful, who was it? Who has died?”
One by one, the employees approached the coffin, and upon looking inside, each was quite surprised. They didn’t understand what they saw.
In the coffin, there was simply a mirror. So when each employee looked in to find out who had been “holding them back from being successful” everyone saw themselves. Next to the mirror, there was a sign that read:
The only person who is able to limit your growth is you. You are the only person who can influence your success. Your life changes when you break through your limiting beliefs and realize that you’re in control of your life. The most influential relationship you can have is the relationship you have with yourself. Now you know who has been holding you back from living up to your true potential. Are you going to keep allowing that person to hold you back?
The Moral:
You can’t blame anyone else if you’re not living up to your potential. You can’t let other people get you down about mistakes you make or their negative perception of your efforts. You have to take personal responsibility for your work–both the good and the bad–and be proactive about making any necessary adjustments.
''',
'''It’s Not That Complicated
The Story:
There was once a very wise man living in ancient times. He was elderly and educated and held knowledge and books to the highest regard.
One day while on a walk, he realized that his shoes were really starting to wear out. Because he spent a lot of time walking on a daily basis, he knew he had to find the best shoes to support and protect his feet. But, back then, this wasn’t such an easy task, as he couldn’t jump online to do some research and have shoes delivered to his door.
The man didn’t want to make things worse by purchasing the wrong shoes and having inadequate protection, which would lead to injuries and the inability to leave his home and walk to find new books to read.
The man gathered all of the books he could that were written by only those that he admired the most to search for the answer to his question, “What do I do if my shoes have fallen apart?”
He read through several books for many hours before finding out that he had no choice but to go buy a new pair of shoes. He then spent a lot of time reading about how to know if a pair of shoes fits properly. Once he was satisfied with the answers he found, he was proud of himself for doing the research and he felt confident in his ability to buy a high-quality replacement for his old shoes. He figured if he hadn’t done his research, he probably would have gone barefoot for the rest of his life, as he had no one to tell him how to fix his shoes.
Following the books’ instructions, the man took a stick and measured his foot with it. He then went to the market and finally came upon a pair of shoes that he liked. However, he realized he had left the stick back at home, which was far away from the shop.
By the time the man returned to the market, the shop was closed. And, by that point, his shoes were completely split, so he had to return home barefoot.
The next morning, he walked back to the market with bare feet, but the shoes that he had chosen the day before had been sold. The wise man explained what had happened to the shopkeeper, who reacted with a sense of surprise, asking, “Why didn’t you buy the shoes yesterday?”
The wise man replied, “Because I forgot the stick that I had used to measure my feet back home. And anyone who knows anything about shoes knows that you have to have the correct measurements of your feet before you can buy shoes. I didn’t want to buy the wrong size, and I was following the normal instructions.”
Even more confused, the shopkeeper asked, “But your foot was with you, why didn’t you just try the shoes on?”
The wise man was equally confused in return and responded, “All the books say shoes must be bought with the exact same measurements of the shoes you already own.”
Laughing, the shop owner replied “Oh, no! You don’t need the advice from books to buy shoes. You just need to have your feet, some money, and some common sense to not complicate things.”
The Moral:
Sometimes you need to take action without overthinking things. Knowledge often comes in handy, but in some circumstances, if you lack experience or common sense, your knowledge will only get you so far. In fact, it could make things seem a lot more complicated than they actually are.
If you’re facing an issue, don’t forget to use your reasoning skills in addition to anything you’ve learned in a formal learning environment.''',
'''Walking on Water
The Story:
Once there was a boy who lived with his family on a farm. They had a beautiful dog who would go down to the pond for hours every day in the spring and summer with the boy to practice retrieving various items. The boy wanted to prepare his dog for any scenario that may come up during duck season because he wanted his dog to be the best hunting dog in the whole county.
The boy and his dog had vigorous training sessions every day until the dog was so obedient, he wouldn’t do anything unless he was told to do so by the boy.
As duck season rolled in with the fall and winter months, the boy and his dog were eager to be at their regular spot down at the pond near their house. Only a few minutes passed before the two heard the first group of ducks flying overhead. The boy slowly raised his gun and shot three times before killing a duck, which landed in the center of the pond.
When the boy signaled his dog to retrieve the duck, the dog charged through the duck blind and bushes toward the pond. However, instead of swimming in the water like he had practiced so many times, the dog walked on the water’s surface, retrieved the duck, and returned it to the boy.  
The boy was astonished. His dog had an amazing ability to walk on water–it was like magic. The boy knew no one would ever believe this amazing thing that he had just witnessed. He had to get someone else down there to see this incredible phenomenon.
The boy went to a nearby farmer’s house and asked if he would hunt with him the next morning. The neighbor agreed, and met up with the boy the following morning at his regular spot by the pond.
The pair patiently waited for a group of ducks to fly overhead, and soon enough, they heard them coming. The boy told the neighbor to go ahead and take a shot, which the neighbor did, killing one duck. Just as the day before, the boy signaled to his dog to fetch the duck. Miraculously, the dog walked on the water again to retrieve the duck.
The boy was bursting with pride and could hardly contain himself when he asked his neighbor, “Did you see that? What do you think?!”
The neighbor responded, “I wasn’t going to say anything, but your dog doesn’t even know how to swim.”
The boy sat in disbelief as his neighbor pointed out a potential flaw of the dog rather than recognizing the fact that what he had just done was a miracle.
The Moral:
People will often downplay others’ abilities or achievements because they’re unable to accomplish the same thing. Don’t let this bring you down. Just move on and keep working on improving yourself. Maintaining a positive mindset is a key part of being successful.
Furthermore, be conscious of instances in which you may be tempted to not give credit where it is deserved. Pointing out other people’s shortcomings does not make you a superior person.
''',
''' The Ultimate Gift
The Story:
There was once a little girl who desperately needed an emergency blood transfusion to save her life.  Her only chance of surviving would be to get a transfusion from her younger brother, who had miraculously overcome the same disease she had, and therefore had antibodies in his blood that were needed to fight the illness.
The doctor explained to the little boy that it would save his sister’s life if he were to give her his blood. The boy hesitated for a moment before agreeing to give his blood if it would help his sister. At the age of 5, this was scary, but he would do anything to save his big sister’s life.
As the blood transfusion was happening, he lay next to his sister in the hospital and was overcome with happiness as he saw the color coming back to her cheeks. Then he looked up at the doctor and quietly asked, “When will I start to die?”

True wealth and happiness aren’t measured by material belongings.
The boy had assumed that he was giving his life in order to save hers. The little boy’s parents were astonished over the misunderstanding that led the boy to think they were choosing his sister over him–and even more astonished that he had agreed to do so.
The doctor replied, explaining that he was not going to die, he was just going to allow his sister to live a long, healthy life alongside him.

The Moral:
This is an example of extreme courage and self-sacrificing love from a young boy that we can all learn from. The love and care that he showed for his sister relays an inspiring message about selflessness. While we may not be faced with such a life or death decision, being selfless in general can help us connect with others, which is rewarding and fulfilling. Selflessness encourages you to act from your heart instead of your ego, and can help fill your life with
''',
'''A Pound is a Pound
The Story:
There was once a farmer who, each week, sold a pound of butter to a baker. After several weeks of buying a pound of butter from the farmer, the baker decided to weigh the butter that he was receiving to ensure it was indeed a full pound. When the baker weighed it, he learned that the butter was under a pound, which enraged him. He felt he was being cheated and he decided to take the farmer to court.

When in court, the judge asked the farmer how he was weighing the butter. The farmer said, “Your Honor, I am poor. I do not own an exact measuring tool. However, I do have a scale.”
The judge then asked if the farmer uses the scale to measure the butter.
The farmer said, “Your Honor, I have been buying a one-pound loaf of bread from the baker since long before he began purchasing butter from me. Whenever the baker brings bread for me, I put it on the scale and then measure out the exact same weight in butter to give him in return. So, if the baker is not getting a pound of butter, he is also not giving a pound of bread like he promised.”
The Moral:
You get what you give. If you try to cheat others out of what you promise them, you will be cheated in return. The more honest you are, the easier it is to trust other people and not suspect they may be cheating you in some way. When you’re honest, not only will other people trust you, but you will also feel more confident in your trust with others. Honesty is always the best route–especially if you want others to be honest with you as well.
''',
'''There Was Once a Boy…

The Story:
There was once a boy who was growing up in a very wealthy family. One day, his father decided to take him on a trip to show him how others lived who were less fortunate. His father’s goal was to help his son appreciate everything that he has been given in life.
The boy and his father pulled up to a farm where a very poor family lived. They spent several days on the farm, helping the family work for their food and take care of their land.
When they left the farm, his dad asked his son if he enjoyed their trip and if he had learned anything during the time they spent with this other family.
The boy quickly replied, “It was fantastic, that family is so lucky!”
Confused, his father asked what he meant by that.
The boy said, “Well, we only have one dog, but that family has four–and they have chickens! We have four people in our home, but they have 12! They have so many people to play with! We have a pool in our yard, but they have a river running through their property that is endless. We have lanterns outside so we can see at night, but they have the wide open sky and the beautiful stars to give them wonder and light. We have a patio, but they have the entire horizon to enjoy–they have endless fields to run around in and play. We have to go to the grocery store, but they are able to grow their own food. Our high fence protects our property and our family, but they don‘t need such a limiting structure, because their friends protect them.”
The father was speechless.

Finally, the boy added, “Thank you for showing me how rich people live, they’re so lucky.”
The Moral:
True wealth and happiness aren’t measured by material belongings. Being around the people you love, enjoying the beautiful, natural environment, and having freedom are much more valuable.
A rich life can mean different things to different people. What are your values and priorities? If you have whatever is important to you, you can consider yourself to be wealthy.
''',
'''Seeking Happiness
The Story:
There were 200 people attending a seminar on mental and physical health. At one point, the speaker told the group they were going to do an activity. He gave each attendee one balloon and told them to write their name on it. Then, the balloons were collected and moved into a very small room.
The participants were then asked to go into the other room and were given 2 minutes to find their balloon.

It was chaos. People were searching frantically for their balloon, pushing each other and running into one another while they grabbed a balloon, looked at it, and inevitably tossed it to the side.
At the end of the 2 minutes, no one had found the balloon that had their name on it.
Then, the speaker asked the participants to go back in the room and pick up one balloon at random, look at the name, and return it to its owner. Within minutes, everyone had been reunited with their original balloon.
The speaker then told the group, “This is what it’s like when people are frantically searching for their own happiness in life. People push others aside to get the things that they want that they believe will bring them happiness. However, our happiness actually lies in helping other people and working together as a community.”
The Moral:
You will get your happiness if you help other people find theirs. The Dalai Lama says, “If you want to be happy, practice compassion.”

Helping others makes us happy because it gives us a sense of purpose. In fact, a study from the London School of Economics found that the more you help other people, the happier you will be. The researchers compared the variance in happiness levels of people who don’t help others on a regular basis to the happiness of weekly volunteers. They found that the participants had the same variance in happiness as those who make $75,000 – $100,000 annually vs $20,000.
Helping others brings us happiness for three reasons:
•	Diversion: When you worry less about your own needs–in this case, finding your own balloon–the stress of that hunt decreases. Taking your focus away from the fact that you can’t find your own balloon lets you divert your attention away from your own problem. The feeling of compassion replaces the feeling of need.
•	Perspective: Having concern for other people helps us remember that we are all facing similar problems in life–no matter what the individual severity of the issue is. Sometimes when we are focused on our own issues, they get put into perspective when we encounter the true suffering of others (for example, bereavement or a severe disability). It’s easy to then realize the excess amount of attention we’ve been giving our own problems. Having compassion helps us put our problems into perspective.
•	Connection: Connecting with others by helping them  can bring happiness into your life. Humans are social beings that need to have positive connections with other people in order to be happy. Connecting with other people enriches our lives and gives us a sense of fulfillment.
''',
'''Cherish Your Struggles
The Story:
One day, a girl came upon a cocoon, and she could tell that a butterfly was trying to hatch. She waited and watched the butterfly struggle for hours to release itself from the tiny hole. All of a sudden, the butterfly stopped moving–it seemed to be stuck.
The girl then decided to help get the butterfly out. She went home to get a pair of scissors to cut open the cocoon. The butterfly was then easily able to escape, however, its body was swollen and its wings were underdeveloped. 
The girl still thought she had done the butterfly a favor as she sat there waiting for its wings to grow in order to support its body. However, that wasn’t happening. The butterfly was unable to fly, and for the rest of its life, it could only move by crawling around with little wings and a large body.
Despite the girl’s good intentions, she didn’t understand that the restriction of the butterfly’s cocoon and the struggle the butterfly had to go through in order to escape served an important purpose. As butterflies emerge from tight cocoons, it forces fluid from their body into their wings to prepare them to be able to fly.

The Moral:
The struggles that you face in life help you grow and get stronger. There is often a reason behind the requirement of doing hard work and being persistent. When enduring difficult times, you will develop the necessary strength that you’ll need in the future.
Without having any struggles, you won’t grow–which means it’s very important to take on personal challenges for yourself rather than relying on other people to always help you. 
''',
'''The Weight of the World
The Story:
Once, a psychology professor walked around his classroom full of students holding a glass of water with his arm straightened out to the side. He asked his students, “How heavy is this glass of water?”
The students started to shout out guesses–ranging anywhere from 4 ounces to one pound.
The professor replied, “The absolute weight of this glass isn’t what matters while I’m holding it. Rather, it’s the amount of time that I hold onto it that makes an impact.
If I hold it for, say, two minutes, it doesn’t feel like much of a burden. If I hold it for an hour, its weight may become more apparent as my muscles begin to tire. If I hold it for an entire day–or week–my muscles will cramp and I’ll likely feel numb or paralyzed with pain, making me feel miserable and unable to think about anything aside from the pain that I’m in.
In all of these cases, the actual weight of the glass will remain the same, but the longer I clench onto it, the heavier it feels to me and the more burdensome it is to hold.

The class understood and shook their heads in agreement.
The professor continued to say, “This glass of water represents the worries and stresses that you carry around with you every day.  If you think about them for a few minutes and then put them aside, it’s not a heavy burden to bear. If you think about them a little longer, you will start to feel the impacts of the stress. If you carry your worries with you all day, you will become incapacitated, prohibiting you from doing anything else until you let them go.”
Don’t carry your worries around with you everywhere you go, as they will do nothing but bring you down.
Put down your worries and stressors. Don’t give them your entire attention while your life is passing you by.
The Moral:

Let go of things that are out of your control. Don’t carry your worries around with you everywhere you go, as they will do nothing but bring you down. Put your “glass down” each night and move on from anything that is unnecessarily stressing you out. Don’t carry this extra weight into the next day.
''',
'''Just Be
The Story:

One evening, after spending several days with his new wife, a man leaned over and whispered into her ear, “I love you.” 
She smiled – and the man smiled back – and she said, “When I’m eighty years old and I’m thinking back on my entire life, I know I will remember this moment.”
A few minutes later, she drifted off to sleep. The man was left with the silence of the room and the soft sound of his wife’s breathing. He stayed awake, thinking about everything they had done together, from their first date to their first vacation together and ultimately to their big wedding. These were just some of the life choices that the couple had made together that had led to this very moment of silence in the presence of each other.
At one point, the man then realized that it didn’t matter what they had done or where they had gone. Nor did it matter where they were going.
The only thing that mattered was the serenity of that very moment.
Just being together. Breathing together. And resting together.
The Moral:
We can’t let the clock, calendar, or pressure from external sources take over our lives and allow us to forget the fact that every moment of our lives is a gift and a miracle – no matter how small or seemingly insignificant it is. Being mindful in the special moments that you spend in the presence of the ones that you love are the moments that truly give your life meaning.
''',
'''Toothpaste Recant
The Story:
One night in July at an all-girls summer camp, the campers were gathered around in a circle for their nighttime devotions. The counselor asked if any of the girls wanted to share something that had happened that day that impacted them. One camper raised her hand and said a girl from another camp cabin had said something that hurt her feelings and she was really upset about it.
The camp counselor went to the bathroom to grab a tube of toothpaste. She took the tube and squeezed it just a bit so some toothpaste came out. She then tried to put the toothpaste back in the tube, but it just created a mess. Then she squeezed the tube even more, pushing more toothpaste out and creating even more of a mess, but none of it would go back into the tube.
The counselor then told the campers, “this toothpaste represents the words you speak. Once you say something that you want to take back, it’s impossible and it only creates a mess. Think before you speak, and make sure your words are going to good use before you let them out.”
The Moral:

Speaking is a fundamental social skill required for living a successful life. However, many are careless with their words, but they hold so much power. They can have a direct impact on the outcome of a situation, creating a helpful or hurtful reaction in our world. The problem is, once words come out of your mouth, no amount of “I’m sorrys” will make them go back in: blurting something out and then attempting to take it back is like shutting the gate after the horse has taken off. 
Thinking before you speak allows you the time to consider the potential impact of your words. Be careful when choosing where and when you let your words out. You can easily hurt other people, and once you do, you can’t take it back.
Think before you speak, and make sure your words are going to good use before you let them out.
Words define who we are by revealing our attitudes and character, giving people an indication of our intellect or ignorance.  Stop for a minute before you speak and question yourself about why you’re saying what you are. Are you trying to relay information? Relate to someone else? Make sure you’re able to take responsibility for whatever you’re about to say.
Moral of the Motivational Story – The Fisherman
You don’t need to wait for tomorrow to be happy and enjoy your life. LIFE is at this moment, live in the present moment and enjoy it fully.
We spend the majority of our lives working, but we seem to forget what we’re working for. As a result, many of us prioritize money, work, and success over family and friends. Such people keep running throughout life for money, power and status. To enjoy your life, you don’t need to be more rich, more powerful and have more status . You need to spend quality time along with your family and best friends. You need to spend time on thing which you love to do.
Happiness lies in small small things of life. Small things and simple things makes you happy. Spending Five more minutes with your kids or family may make you happy which you won’t get from thousands of dollars. Don’t Chase Happiness. Recognize It. Happiness is a journey, not a destination. You need to find out the same with in you.
Well, it does not mean that money and hard work is not important. Both of these are a necessary part of life, and the fisherman is evidently a hard worker. But rather than making work his life, the fisherman treasures his family and friends. He has made fair work life balance.''',
'''An Old Man Lived in the Village
Like An old man lived in the village. The whole village was tired of him; he was always gloomy, he constantly complained and was always in a bad mood. The longer he lived, the viler he became and more poisonous were his words. People did their best to avoid him because his misfortune was contagious. He created the feeling of unhappiness in others. But one day, when he turned eighty, an incredible thing happened. Instantly everyone started hearing the rumor: “The old man is happy today, he doesn’t complain about anything, smiles, and even his face is freshened up.” The whole village gathered around the man and asked him, “What happened to you?” The old man replied, “Nothing special. Eighty years I’ve been chasing happiness and it was useless. And then I decided to live without happiness and just enjoy life. That’s why I’m happy now.”
Moral of the story: Don’t chase happiness. Enjoy your life.
''',
'''The Wise Man
Like People visit a wise man complaining about the same problems over and over again. One day, he decided to tell them a joke and they all roared with laughter. After a few minutes, he told them the same joke and only a few of them smiled. Then he told the same joke for a third time, but no one laughed or smiled anymore. The wise man smiled and said: “You can’t laugh at the same joke over and over. So why are you always crying about the same problem?”
 Moral of the story: Worrying won’t solve your problems, it’ll just waste your time and energy.''',
'''Having a Best Friend
Like Two friends were walking through the desert. At one stage in their journey, they had an argument and one friend slapped the other one in the face. The one who got slapped was hurt, but without saying anything he wrote in the sand, “Today my best friend slapped me in the face.” They kept on walking until they found an oasis, where they decided to have a wash. The one who had been slapped got stuck in a mire and started drowning, but his friend saved him. After he had recovered from his shock, he wrote on a stone, “Today my best friend saved my life.” The friend who slapped and saved his best friend asked him, “After I hurt you, you wrote in the sand and now, you write in stone, why?” The other friend replied, “When someone hurts us we should write it down in sand where winds of forgiveness can erase it away. But, when someone does something good for us, we must engrave it in stone where no wind can ever erase it.”
 Moral of the story: Don’t value the things you have in your life. Value those who you have in your life.'''

]

story_labels = ["safe alone strong fear surprise","safe strong fearless happy attracted","anxious strong happy attracted",
                "surprise safe happy",
                "safe happy","strong fear surprise sad happy","safe angry respect loved strong lustful charming surprise sad happy admire",
                "surprise happy attracted","fearful overcome","nothing","angry","apathetic","happy attracted",
                "sad","fearful happy excited help","angry alone hated sad embarrassed","nothing","nothing",
                "strong fearful angry alone","nothing","alone fearful happy hated","attracted",
                "happy","alone sad motivation","happy embarrassed","sad obsessed happy charming loved","sad surprise",
                "sad loved adequate","nothing","nothing","attracted fearful happy help excited","nothing","strong angry fearful happy alone",
                "surprise","respect attached happy confident surprise","surprise attracted","surprise attracted overcome motivated",
                "cheated angry confident","happy surprise ecstatic","happy focused sad","happy","attached apathetic","belittled",
                "sad motivation fearful happy powerless","happy sad","nothing","sad"
                ]

pos_text =  open('positive emotions.txt',encoding='utf-8').read()
neg_text =  open('negative emotion.txt',encoding='utf-8').read()

 #1st argument the string need to replace ,2nd is the argument
# use replacement , 3rd the argument need to delete

pos_clean_text = pos_text.translate(str.maketrans('','',string.punctuation))
neg_clean_text = neg_text.translate(str.maketrans('','',string.punctuation))


def pos_neg_sentiment_analyser(emotion_list):
    pos_emotion_list = []
    neg_emotion_list = []
    with open('positive emotions.txt', encoding='utf-8') as file:
        for line in file:
            clear_line = line.replace("\n", "")
            l = clear_line.split(',')
            for word in l:
                if word in emotion_list:
                    pos_emotion_list.append(word)


    with open('negative emotion.txt', encoding='utf-8') as file:
        for line in file:
            clear_line = line.replace("\n", "")
            l=clear_line.split(',')
            for word in l:
                if word in emotion_list:
                    neg_emotion_list.append(word)

    neutral_emotion = []

    for emotion in emotion_list:
        if emotion not in neg_emotion_list:
            if emotion not in pos_emotion_list:
                neutral_emotion.append(emotion)
    return pos_emotion_list,neg_emotion_list,neutral_emotion

#STORY ANALYSER




def sentiment_analyse(text):
    text = text.lower()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))

    temp_words = word_tokenize(clean_text, 'english')
    emotion_list = []
    final_words = []
    for words in temp_words:
        if words not in stopwords.words():
            final_words.append(words)
    with open('emotion.txt', encoding='utf-8') as file:
        for line in file:
            clear_line = line.replace("\n", "").replace("'", "").replace(",", "").replace(' ', '').strip()
            word, emotion = clear_line.split(":")

            for w in final_words:
                if word == w:
                    emotion_list.append(emotion)

    emotion_list = set(emotion_list)
    emotion_list = list(emotion_list)

    emo = ""
    for i in range(len(emotion_list)):
        emo = emo+" "+emotion_list[i]
    global pos_text
    global neg_text
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    nlp_pos =  score['pos']
    nlp_neg =  score['neg']
    nlp_comp = score['compound']
    opinion = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    textblob_sentiment_pos = opinion.sentiment[1]
    textblob_sentiment_neg = opinion.sentiment[2]

    pos_sentiment = [clean_text,pos_clean_text]
    neg_sentiment = [clean_text,neg_clean_text]
    cv = CountVectorizer()

    count_matrix_pos = cv.fit_transform(pos_sentiment)

    count_matrix_neg = cv.fit_transform(neg_sentiment)

    pos_sklearn = cosine_similarity(count_matrix_pos)[0][1]

    neg_sklearn = cosine_similarity(count_matrix_neg)[0][1]
    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(text)  # text can be tokenized by other ways
    score2 = hiv4.get_score(tokens)
    ps_pos_hiv = (score2['Positive']/100)
    ps_neg_hiv = (score2['Negative']/100)
    lm = ps.LM()
    tokens = lm.tokenize(text)
    score1 = lm.get_score(tokens)
    ps_pos_lm = (score1['Positive']/100)
    ps_neg_lm  = (score1['Negative']/100)
    text_emotion = te.get_emotion(text)
    text_emotion_pos = 0
    text_emotion_neg = 0
    if(text_emotion['Happy']>text_emotion['Sad']):
        text_emotion_pos = text_emotion['Happy']+text_emotion['Surprise']
        text_emotion_neg = text_emotion['Sad']+text_emotion['Angry']+text_emotion['Fear']

    else:
        text_emotion_pos = text_emotion['Happy']
        text_emotion_neg = text_emotion['Sad'] + text_emotion['Angry'] + text_emotion['Fear']+text_emotion['Surprise']

    positive_mentality = (nlp_pos+2*textblob_sentiment_pos+pos_sklearn+2*ps_pos_lm+2*ps_pos_hiv+text_emotion_pos)
    negative_mentality = (nlp_neg+2*textblob_sentiment_neg+neg_sklearn+2*ps_neg_lm+2*ps_neg_hiv+text_emotion_neg)
    afinn = Afinn(language='en')
    sent_point = afinn.score(text) / 100
    if sent_point>0:
        positive_mentality = positive_mentality + sent_point
        positive_mentality/=12
    else:
        negative_mentality = negative_mentality-sent_point
        negative_mentality/=12
    neutral_mentality = (score['neu']+score2['Polarity']+score1['Polarity'])/3
    subjectivity =  score1['Subjectivity']+score2['Subjectivity']/2

    expression= []
    value =[]

    pos_emotions, neg_emotions, neutral_emotion=pos_neg_sentiment_analyser(emotion_list)
    if len(emotion_list) > 0:
        emotions = Counter(emotion_list)
        for key in emotions.keys():
            if key in pos_emotions:
                expression.append(key)
                value.append(emotions[key]*positive_mentality)

        for key in emotions.keys():
            if key in neg_emotions:
                expression.append(key)
                value.append(emotions[key] * negative_mentality)
        for key in emotions.keys():
            if key in neutral_emotion:
                expression.append(key)
                value.append(emotions[key] * neutral_mentality)

    if len(emotion_list) == 0:
        if positive_mentality - negative_mentality > 0.025 or negative_mentality - positive_mentality > 0.025:
            print("Patient have a neutral statement")
            text_emotion = te.get_emotion(text)
            print(text_emotion)
            print("Patient neutral emotion point is : ", neutral_mentality)
            print(subjectivity)


    video = []
    for i in range(len(expression)):
        for j in range(i+1,len(value)):
            if value[i]<value[j]:
                a = expression[i]
                expression[i] = expression[j]
                expression[j] = a
    expression = expression[:2]
    for key in suggestions.keys():
        if key in expression:
            video.append(suggestions[key])


    #STORY
    story = []
    max_story = -1000
    label=0
    for i in range(len(story_labels)):
        story_sentiment = [str(story_labels[i]), emo]


        count_matrix_story = cv.fit_transform(story_sentiment)
        story_similarity = cosine_similarity(count_matrix_story)[0][1]
        if (max_story < story_similarity):
            max_story = story_similarity
            label = i

    story.append(stories[label])


    #quotes

    quotes = []
    with open("quotes.txt", encoding="utf-8") as file:
        for line in file:

            try:
                quote, author = line.split("-")
                quote = quote[3:]
                quotes.append(quote)
            except:
                continue
    motivational_quotes = []
    for i in range(2):
        motivational_quotes.append(r.choice(quotes).lower())

    articles = []

    with open("article.txt", encoding="utf-8") as file:
        for line in file:

            try:

                article = line[3:]
                articles.append(article)
            except:
                continue

    motivational_article = []
    for i in range(2):
        motivational_article.append(r.choice(articles).lower())

    playists = []
    with open("playlist.txt", encoding="utf-8") as file:
        for line in file:

            try:
                if line !="":
                    playlist = line[3:]
                    playists.append(playlist)
            except:
                continue

    motivational_song = []
    for i in range(2):
        motivational_song.append(r.choice(playists))
    solution = motivational_quotes+ video + motivational_song +  motivational_article + story


    return solution

'''from tkinter import *

root= Tk()
root.title("chatbot")
root.geometry('400x500')

#create a chat window
chatWindow = Text(root,bd=1,bg='black', width=50,height=8)
chatWindow.place(x=6,y=6,height=385,width=370)

#create the text message

messageWindow = Text(root,bg='black',width=30,height=4)
messageWindow.place(x=128,y=400,height=88,width=260)

#create a button to send the message

Button = Button(root,bg='blue',text='Send',activebackground='light blue',width=12,height=5)
Button.place(x=6,y=400,height=88,width=120)

root.mainloop()'''
#software
question = ["What's thought coming to your mind after you tested covid positive","what feeling coming to your mind ","are you finding your family supportive for you ?"]
from tkinter import *
root = Tk()
answer =""
index = 0
def send():
    global answer
    global index
    index += 1
    if(index==3):
        sol  = sentiment_analyse(answer)
        txt.insert(END,"\n"+"BOT's solution : \n"+str(sol))
    else:
        answer = answer +" "+e.get()
        txt.insert(END,"\n"+"Patient : "+e.get())
        txt.insert(END,"\n"+"Bot :"+question[index])
txt = Text(root)
txt.grid(row=0,column=0,columnspan=2)
e = Entry(root,width=100)
send = Button(root,text="Send",command=send).grid(row=1,column=1)
e.grid(row=1,column=0)
root.title("CHATBOT")
txt.insert(END,"\n"+"Bot : "+question[index])
root.mainloop()



# print(sentiment_analyse("I am Happy"))


'''video,quotes,story,article,song = sentiment_analyse("at first i was afraid but now i am confident that can overcome everything")
print("videos")
for i in range(len(video)):
    print(video[i])
print("quotes")
for i in range(len(quotes)):
    print(quotes[i])
print("story")
for i in range(len(story)):
    print(story[i])

print("article")
for i in range(len(article)):
    print(article[i])

print("song")
for i in range(len(song)):
    print(song[i])'''









