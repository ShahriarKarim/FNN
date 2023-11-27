import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the datasets
nba_data = pd.read_csv('NBA_API_DATA.csv')
player_mvp_stats = pd.read_csv('player_mvp_stats.csv')

team_mapping = {
    'LAL': 'LAL', 'PHO': 'PHO', 'DAL': 'DAL', 'MIA': 'MIA', 'CLE': 'CLE', 'WSB': 'WSB', 'CHI': 'CHI', 'GSW': 'GSW',
    'IND': 'IND', 'WAS': 'WAS', 'MIN': 'MIN', 'BOS': 'BOS', 'HOU': 'HOU', 'DEN': 'DEN', 'ORL': 'ORL', 'NOH': 'NOP',
    'TOR': 'TOR', 'SAC': 'SAC', 'CHO': 'CHO', 'POR': 'POR', 'DET': 'DET', 'PHI': 'PHI', 'UTA': 'UTA', 'MIL': 'MIL',
    'VAN': 'MEM', 'SEA': 'OKC', 'NJN': 'BRK', 'NOK': 'NOP', 'LAC': 'LAC', 'OKC': 'OKC', 'ATL': 'ATL', 'CHA': 'CHO',
    'MEM': 'MEM', 'NYK': 'NYK', 'NOP': 'NOP', 'BRK': 'BRK', 'SAS': 'SAS', 'CHH': 'CHO'
}

player_mvp_stats['Tm'] = player_mvp_stats['Tm'].map(team_mapping)

combined_data = pd.merge(nba_data, player_mvp_stats, left_on='team', right_on='Tm', how='inner')
# Get all player names

unique_players = combined_data['Player'].unique()

pd.set_option('display.max_rows', None)

print(unique_players)

pd.reset_option('display.max_rows')

# Sample mapping of player names to their corresponding weights
player_weights = {
    'Nikola Jokic': 0.98,
    'Joel Embiid': 0.97,
    'Stephen Curry': 0.97,
    'Luka Doncic': 0.96,
    'Giannis Antetokounmpo': 0.96,
    'LeBron James': 0.96,
    'Kevin Durant': 0.96,
    'Jayson Tatum': 0.95,
    'Devin Booker': 0.94,
    'Jimmy Butler': 0.94,
    'Donovan Mitchell': 0.93,
    'Damian Lillard': 0.93,
    'Anthony Davis': 0.93,
    'Shai Gilgeous-Alexander': 0.93,
    'Kawhi Leonard': 0.92,
    'Ja Morant': 0.92,
    'Anthony Edwards': 0.90,
    'Tyrese Haliburton': 0.90,
    'Zion Williamson': 0.89,
    'James Harden': 0.89,
    'Kyrie Irving': 0.89,
    'Paul George': 0.89,
    'Jaylen Brown': 0.89,
    'Trae Young': 0.88,
    'Jamal Murray': 0.88,
    'Jalen Brunson': 0.88,
    'De’Aaron Fox': 0.88,
    'Bam Adebayo': 0.87,
    'Tyrese Maxey': 0.87,
    'Kristaps Porzingis': 0.87,
    'Victor Wembanyama': 0.86,
    'Scottie Barnes': 0.86,
    'Bradley Beal': 0.86,
    'Dejounte Murray': 0.86,
    'Jaren Jackson Jr.': 0.86,
    'Domantas Sabonis': 0.86,
    'Lauri Markkanen': 0.86,
    'Jrue Holiday': 0.86,
    'Mikal Bridges': 0.86,
    'Evan Mobley': 0.85,
    'Cade Cunningham': 0.85,
    'LaMelo Ball': 0.85,
    'Darius Garland': 0.85,
    'Karl-Anthony Towns': 0.85,
    'DeMar DeRozan': 0.85,
    'Pascal Siakam': 0.85,
    'Tyler Herro': 0.85,
    'C.J. McCollum': 0.85,
    'Khris Middleton': 0.85,
    'Desmond Bane': 0.85,
    'Paolo Banchero': 0.84,
    'Rudy Gobert': 0.84,
    'Zach LaVine': 0.84,
    'Brandon Ingram': 0.84,
    'Aaron Gordon': 0.84,
    'Nicolas Claxton': 0.84,
    'Kyle Kuzma': 0.84,
    'Alperen Sengun': 0.84,
    'Klay Thompson': 0.83,
    'Chet Holmgren': 0.83,
    'Draymond Green': 0.83,
    'R.J. Barrett': 0.83,
    'Jarrett Allen': 0.83,
    'Michael Porter Jr.': 0.83,
    'Julius Randle': 0.83,
    'Anfernee Simons': 0.83,
    'Jalen Duren': 0.83,
    'Derrick White': 0.83,
    'Malcolm Brogdon': 0.83,
    'Myles Turner': 0.83,
    'Chris Paul': 0.82,
    'Jalen Green': 0.82,
    'Shaedon Sharpe': 0.82,
    'Ausar Thompson': 0.82,
    'Deandre Ayton': 0.82,
    'Franz Wagner': 0.82,
    'OG Anunoby': 0.82,
    'Fred VanVleet': 0.82,
    'Jonas Valanciunas': 0.82,
    'Nikola Vucevic': 0.82,
    'Tobias Harris': 0.82,
    'Jalen Williams': 0.82,
    'Marcus Smart': 0.82,
    'Malik Monk': 0.82,
    'Brook Lopez': 0.82,
    'Bobby Portis': 0.82,
    'Mitchell Robinson': 0.82,
    'Cam Thomas': 0.82,
    'Devin Vassell': 0.82,
    'Bojan Bogdanovic': 0.82,
    'Austin Reaves': 0.82,
    'Josh Giddey': 0.81,
    'Lonzo Ball': 0.81,
    'Jordan Poole': 0.81,
    'Russell Westbrook': 0.81,
    'Robert Williams III': 0.81,
    'Clint Capela': 0.81,
    'John Collins': 0.81,
    'Jerami Grant': 0.81,
    'Onyeka Okongwu': 0.81
}

# Add player weights to combined_data
combined_data['player_weight'] = combined_data['Player'].map(player_weights)

default_weight = 0.5

combined_data['player_weight'] = combined_data['Player'].map(player_weights).fillna(default_weight)

team_weights = {
  'pts': 0.833,
  'fg%_max': 1.0,
  '3p%_max': 0.667,
  'ft%_max': 0.417,
  'trb_max': 1.0,
  'ast_max': 0.917,
  'stl_max': 0.75,
  'blk_max': 0.75
}

for col, weight in team_weights.items():
    combined_data[col] = combined_data[col] * weight

# Encoding teams
combined_data['home_team'] = combined_data.apply(lambda row: row['team'] if row['home_opp'] == 0 else row['team_opp'], axis=1)
combined_data['away_team'] = combined_data.apply(lambda row: row['team_opp'] if row['home_opp'] == 0 else row['team'], axis=1)

home_teams_encoded = pd.get_dummies(combined_data['home_team'], prefix='home')
away_teams_encoded = pd.get_dummies(combined_data['away_team'], prefix='away')
combined_data = pd.concat([combined_data, home_teams_encoded, away_teams_encoded], axis=1)

# Preparing data for training
X = combined_data.drop(['won'], axis=1)
y = combined_data['won'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network model
num_features = X_train_scaled.shape[1]
input_layer = tf.keras.layers.Input(shape=(num_features,))
hidden1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
dropout1 = tf.keras.layers.Dropout(0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(0.3)(hidden2)
hidden3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=1, batch_size=4, validation_data=(X_test_scaled, y_test))

# Prediction function
def predict_winner(home_team, away_team, model, scaler, combined_data):
    # Prepare the input data
    game_data = combined_data.head(1).copy()  # Copy the structure
    for column in combined_data.columns:
        if column.startswith('home_') or column.startswith('away_'):
            game_data[column] = 0  # Reset team columns
    game_data[f'home_{home_team}'] = 1
    game_data[f'away_{away_team}'] = 1

    # Scale the features
    non_team_columns = [col for col in combined_data.columns if not col.startswith('home_') and not col.startswith('away_')]
    game_data_scaled = scaler.transform(game_data[non_team_columns])

    # Predict
    prediction = model.predict(game_data_scaled)
    print(prediction)
    return prediction

# Using the function
home_team = 'LAL'  # Home team
away_team = 'RAP'  # Away team
prediction = predict_winner(home_team, away_team, model, scaler, combined_data)

# Predicted winner
predicted_winner = home_team if prediction[0][0] > 0.5 else away_team
print("Predicted winner: {predicted_winner}")