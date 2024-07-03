from project import summarize_text, time_function
from openai import OpenAI
import time


client = OpenAI()

txt ="All right, welcome, everyone, to an introduction to artificial intelligence with Python. My name is Brian Yu. And in this class, we'll explore some of the ideas and techniques and algorithms that are at the foundation of artificial intelligence. Now, artificial intelligence covers a wide variety of types of techniques. Anytime you see a computer do something that appears to be intelligent or rational in some way, like recognizing someone's face in a photo, or being able to play a game better than people can, or being able to understand human language when we talk to our phones and they understand what we mean and are able to respond back to us, these are all examples of AI, or artificial intelligence. And in this class, we'll explore some of the ideas that make that AI possible. So we'll begin our conversations with search, the problem of we have an AI, and we would like the AI to be able to search for solutions to some kind of problem, no matter what that problem might be, whether it's trying to get driving directions from point A to point B, or trying to figure out how to play a game, given a tic-tac-toe game, for example, figuring out what move it ought to make. After that, we'll take a look at knowledge. Ideally, we want our AI to be able to know information, to be able to represent that information, and more importantly, to be able to draw inferences from that information, to be able to use the information it knows and draw additional conclusions. So we'll talk about how AI can be programmed in order to do just that. Then we'll explore the topic of uncertainty, talking about ideas of what happens if a computer isn't sure about a fact, but maybe is only sure with a certain probability. So we'll talk about some of the ideas behind probability and how computers can begin to deal with uncertain events in order to be a little bit more intelligent in that sense as well. After that, we'll turn our attention to optimization, problems of when the computer is trying to optimize for some sort of goal, especially in a situation where there might be multiple ways that a computer might solve a problem, but we're looking for a better way, or potentially the best way, if that's at all possible. Then we'll take a look at machine learning, or learning more generally, and looking at how, when we have access to data, our computers can be programmed to be quite intelligent by learning from data and learning from experience, being able to perform a task better and better based on greater access to data. So your email, for example, where your email inbox somehow knows which of your emails are good emails and which of your emails are spam, these are all examples of computers being able to learn from past experiences and past data. We'll take a look, too, at how computers are able to draw inspiration from human intelligence, looking at the structure of the human brain and how neural networks can be a computer analog to that sort of idea, and how by taking advantage of a certain type of structure of a computer program, we can write neural networks that are able to perform tasks very, very effectively. And then finally, we'll turn our attention to language, not programming languages, but human languages that we speak every day, and taking a look at the challenges that come about as a computer tries to understand natural language and how it is that some of the natural language processing that occurs in modern artificial intelligence can actually work. But today, we'll begin our conversation with search, this problem of trying to figure out what to do when we have some sort of situation that the computer is in, some sort of environment that an agent is in, so to speak, and we would like for that agent to be able to somehow look for a solution to that problem. Now, these problems can come in any number of different types of formats. One example, for instance, might be something like this classic 15 puzzle with the sliding tiles that you might have seen, where you're trying to slide the tiles in order to make sure that all the numbers line up in order. This is an example of what you might call a search problem. The 15 puzzle begins in an initially mixed up state, and we need some way of finding moves to make in order to return the puzzle to its solved state. But there are similar problems that you can frame in other ways. Trying to find your way through a maze, for example, is another example of a search problem. You begin in one place. You have some goal of where you're trying to get to, and you need to figure out the correct sequence of actions that will take you from that initial state to the goal. And while this is a little bit abstract, anytime we talk about maze solving in this class, you can translate it to something a little more real world, something like driving directions. If you ever wonder how Google Maps is able to figure out what is the best way for you to get from point A to point B and what turns to make at what time depending on traffic, for example, it's often some sort of search algorithm. You have an AI that is trying to get from an initial position to some sort of goal by taking some sequence of actions. So we'll start our conversations today by thinking about these types of search problems and what goes in to solving a search problem like this in order for an AI to be able to find a good solution. In order to do so, though, we're going to need to introduce a little bit of terminology, some of which I've already used. But the first term we'll need to think about is an agent. An agent is just some entity that perceives its environment. It somehow is able to perceive the things around it and act on that environment in some way. So in the case of the driving directions, your agent might be some representation of a car that is trying to figure out what actions to take in order to arrive at a destination. In the case of the 15 puzzle with the sliding tiles, the agent might be the AI or the person that is trying to solve that puzzle, to try and figure out what tiles to move in order to get to that solution. Next, we introduce the idea of a state. A state is just some configuration of the agent in its environment. So in the 15 puzzle, for example, any state might be any one of these three, for example. A state is just some configuration of the tiles. And each of these states is different and is going to require a slightly different solution. A different sequence of actions will be needed in each one of these in order to get from this initial state to the goal, which is where we're trying to get. So the initial state, then. What is that? The initial state is just the state where the agent begins. It is one such state where we're going to start from. And this is going to be the starting point for our search algorithm, so to speak. We're going to begin with this initial state and then start to reason about it, to think about what actions might we apply to that initial state in order to figure out how to get from the beginning to the end, from the initial position to whatever our goal happens to be. And how do we make our way from that initial position to the goal? Well, ultimately, it's via taking actions. Actions are just choices that we can make in any given state. And in AI, we're always going to try to formalize these ideas a little bit more precisely, such that we could program them a little bit more mathematically, so to speak. So this will be a recurring theme. And we can more precisely define actions as a function. We're going to effectively define a function called actions that takes an input, s, where s is going to be some state that exists inside of our environment. And actions of s is going to take the state as input and return as output the set of all actions that can be executed in that state. And so it's possible that some actions are only valid in certain states and not in other states. And we'll see examples of that soon, too. So in the case of the 15 puzzle, for example, there are generally going to be four possible actions that we can do most of the time. We can slide a tile to the right, slide a tile to the left, slide a tile up, or slide a tile down, for example. And those are going to be the actions that are available to us. So somehow our AI, our program, needs some encoding of the state, which is often going to be in some numerical format, and some encoding of these actions. But it also needs some encoding of the relationship between these things. How do the states and actions relate to one another? And in order to do that, we'll introduce to our AI a transition model, which will be a description of what state we get after we perform some available action in some other state. And again, we can be a little bit more precise about this, define this transition model a little bit more formally, again, as a function. The function is going to be a function called result that this time takes two inputs. Input number one is s, some state. And input number two is a, some action. And the output of this function result is it is going to give us the state that we get after we perform action a in state s. So let's take a look at an example to see more precisely what this actually means. Here's an example of a state of the 15 puzzle, for example. And here's an example of an action, sliding a tile to the right. What happens if we pass these as inputs to the result function? Again, the result function takes this board, this state, as its first input. And it takes an action as a second input. And of course, here I'm describing things visually so that you can see visually what the state is and what the action is. In a computer, you might represent one of these actions as just some number that represents the action. Or if you're familiar with enums that allow you to enumerate multiple possibilities, it might be something like that. And this state might just be represented as an array or two-dimensional array of all of these numbers that exist. But here, we're going to show it visually just so you can see it. But when we take this state and this action, pass it into the result function, the output is a new state, the state we get after we take a tile and slide it to the right. And this is the state we get as a result. If we had a different action and a different state, for example, and pass that into the result function, we'd get a different answer altogether. So the result function needs to take care of figuring out how to take a state and take an action and get what results. And this is going to be our transition model that describes how it is that states and actions are related to each other. If we take this transition model and think about it more generally and across the entire problem, we can form what we might call a state space, the set of all of the states we can get from the initial state via any sequence of actions by taking 0 or 1 or 2 or more actions in addition to that. So we could draw a diagram that looks something like this, where every state is represented here by a game board. And there are arrows that connect every state to every other state we can get to from that state. And the state space is much larger than what you see just here. This is just a sample of what the state space looks like."

summary = time_function(summarize_text, "remote chat", english_txt = txt, use_local = False)


summary = time_function(summarize_text, "local chat", english_txt = txt , use_local = True)