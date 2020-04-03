from data_processing import normalize, decompose
from io_routines import readCSV, saveNetwork


nao = readCSV("NAO.csv")
human = readCSV("HUMAN.csv")

