This repository contains code for the NBA sentiment project decribed in https://medium.com/@io.1610/ripple-effect-sentiment-contagion-on-nba-reddit-e0c249a97d62.

To run the code use notebook_nba.ipynb

There are a couple of data files in the repository.
flair_to_team_dict.pickle: Dictionary mapping reddit flairs to a team
lotr_characters_data2.csv: Contains character names from Lord of the Ring (https://github.com/juandes/lotr-names-classification/blob/master/characters_data.csv). 
  This is used to create conversation trees to feed into the language model without risking name overlap in the text.
nba_schedule_2022.txt: 2022 NBA playoff schedule including results.
