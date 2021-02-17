# league-draft-assistant
A machine learning based character selection recommendation system for the popular online game League of Legends (LoL). Written entirely in Python with PyTorch for learning.

# What is League of Legends?
From the developer Riot Games:
>League of Legends is a Multiplayer Online Battle Arena (MOBA) that blends the speed and intensity of an RTS with RPG elements. Two teams of powerful champions, each with unique designs and play styles, battle head-to-head across multiple maps and game modes. With an ever-expanding roster of champions, frequent updates, and a thriving tournament scene, League of Legends offers endless replay-ability for players of every skill level.
>

# Why league-draft-assistant?
Inherent to a team-based game with a ranked leaderboard, a high skill-ceiling, and a large playerbase, League is an extremely competitive game.
At the highest levels of competition, either individual ranked play dubbed "Solo Queue" or simply "soloq" by players (and here on out referred to as such) or organized eSports leagues and tournaments with millions of dollars in prize earnings, players need every advantage they can get.

Before the start of every match there is a strategic "drafting" phase where players are able to ban specific champions from the enemy team and pick their own champions for the match, taking turns in phases.
During this champion selection phase, each team can gain significant advantages through picking champions that synergize well together and are naturally strong against the champions that the enemy team has picked.
This drafting phase is so consequential to a how an individual match will play out that professional teams have large amounts of staff and coaches that dedicated to analyzing how champions interact with each other in-game and researching opposing teams such that they can strategize what champions they should pick for every match.

The goal of league-draft-assistant is to gather and analyze data from a large amount of matches and give recommendations to players and analysts about what champions should be picked in order to give their team the largest advantage and have the highest probability of winning.
