#Truong_Nguyen_Deep_Learning_for_Predicting_De_novo_Promoters

import pandas as pd
import numpy as np
import csv, os
from numpy import savetxt
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier

TSS = pd.read_csv('de_novo_promoters_traindata.csv')

X_train = TSS.iloc[:, :-1]  # Exclude the last column
y_train = TSS.iloc[:, -1]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

X = X_train.dropna(axis='columns')

def create_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_feature_scores(X_train, y_train):
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    feature_scores = np.abs(model.get_weights()[0]).sum(axis=1)
    return feature_scores


def initilization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[:int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    newtp = []
    newfp = []
    newtn = []
    newfn = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for chromosome in population:
        tp = []
        fp = []
        tn = []
        fn = []
        acc = []
        for train, test in kfold.split(X, y_train):
            model = RandomForestClassifier()
            model.fit(X.iloc[train, chromosome], y_train[train])
            true_labels = np.asarray(y_train[test])
            predictions = model.predict(X.iloc[test, chromosome])

            ntp, nfn, ntn, nfp = confusion_matrix(true_labels, predictions).ravel()
            tp.append(ntp)
            fp.append(nfp)
            tn.append(ntn)
            fn.append(nfn)
            acc.append(accuracy_score(true_labels, predictions))

        scores.append(np.mean(acc))
        newtp.append(np.sum(tp))
        newfp.append(np.sum(fp))
        newtn.append(np.sum(tn))
        newfn.append(np.sum(fn))

    scores, population = np.array(scores), np.array(population)

    weights = scores / np.sum(scores)
    newtp, newfp, newtn, newfn = np.array(newtp), np.array(newfp), np.array(newtn), np.array(newfn)
    inds = np.argsort(scores)

    return (
        list(scores[inds][::-1]),
        list(population[inds, :][::-1]),
        list(weights[inds][::-1]),
        list(newtp[inds][::-1]),
        list(newfp[inds][::-1]),
        list(newtn[inds][::-1]),
        list(newfn[inds][::-1]),
    )

def selection(pop_after_fit, weights, k):
    pop_after_sel = []
    selected_pop = random.choices(pop_after_fit, weights=weights, k=k)
    for t in selected_pop:
        pop_after_sel.append(t)
    return pop_after_sel

def crossover(p1, p2, crossover_rate):
    
    c1, c2 = p1.copy(), p2.copy()
    
    if random.random() < crossover_rate:
        
        pt = random.randint(1, len(p1) - 2)
       
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))
    return [c1, c2]

def mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
       
        if random.random() < mutation_rate:
           
            chromosome[i] = not chromosome[i]

def generations(size, n_feat, crossover_rate, mutation_rate, n_gen):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)

    for i in range(n_gen):
        scores, pop_after_fit, weights, tp, fp, tn, fn = fitness_score(population_nextgen)
        score = scores[0]
        print('gen', i, score)

        k = size - 2
        pop_after_sel = selection(pop_after_fit, weights, k)

        
        children = []
        for i in range(0, len(pop_after_sel), 2):
            
            p1, p2 = pop_after_sel[i], pop_after_sel[i + 1]
           
            for c in crossover(p1, p2, crossover_rate):
                mutation(c, mutation_rate)
                
                children.append(c)

       
        pop_after_mutated = children
        population_nextgen = []
        for c in pop_after_fit[:2]:
            population_nextgen.append(c)
        for p in pop_after_mutated:
            population_nextgen.append(p)

        best_chromo.append(pop_after_fit[0])
        best_score.append(score)

    return best_chromo, best_score


# Running Genetic Algorithm
best_chromo, best_score = generations(size=50, n_feat=X.shape[1], crossover_rate=0.8, mutation_rate=0.05, n_gen=1000)
print("Best Chromosome:", best_chromo)
print("Best Score:", best_score)
