"Copyright [2019] Gavan Adrian-George, 334CA"

Nume, prenume: Gavan, Adrian-George

Grupa, seria: 334CA

Tema 3 ASC - Parallel hashtable

A. Idee generala:
=================

- Clasa pentru hashtable a fost gandita sa contina 4 campuri, anume: limita maxima de elemente
ce pot incapea in hashtable, numarul total de elemente din hashtable si 2 vectori, unul pentru
chei si altul pentru valori.
- Am ales sa folosesc 2 vectori si nu un vector de structuri pentru ca imi este mult mai usor
la functia de reshape sa lucrez cu 2 vectori. Se putea utiliza si un vector de structuri.
Initial foloseam acelasi kernel de insert atat pentru insert cat si pentru reshape si era
mult mai optim cu 2 vectori. Intre timp am modificat codul si am facut kenel-uri separate, dar
am lasat tot 2 vectori pentru a nu modifica tot codul.
- Toate datele hashtable-ului sunt stocate in VRAM.
- Implementarea este o combinatie de linear probing si Cuckoo based. Se utilizeaza 3 functii
de hash. Se calculeaza o pozitie cu prima functie de hash. Daca nu este buna, se utilizeaza
a doua functie de hash. Daca nici aceasta pozitie nu este buna se utilizeaza a 3-a functie de
hash. Daca nici ultima pozitie nu este buna, folosim linear probing si ne ducem la "dreapta"
pana cand gasim o pozitie corespunzatoare.
- S-au utilizat kernel-uri pentru insertBatch, getBatch si reshape.

B. Continut arhiva:
===================

- bench.py, std_hashtable.cpp si test_map.cpp - fisierele din schelet pe care nu trebuie sa le
modificam.
- script.sh - script ce poate fi utilizat pentru testarea pe cluster
- gpu_hashtable.cpp si gpu_hashtable.hpp - contin implementarea hashtable-ului
- Makefile - utilizat pentru crearea executabilului (s-a mai adaugat si dependinta utils.hpp)
- utils.hpp - fisier nou adaugat in care a fost mutata functia DIE si vectorul de numere prime.
- Results.png - imagine cu una din rularile de pe cluster

C. Rulare program:
==================

- Programul a fost rulat pe coada ibm-dp. Coada hp-sl nu functiona in momentul cand lucram eu
la tema.
- Ma logam pe cluster.
- Intram pe coada ibm-dp cu comanda qlogin -q ibm-dp.q
- Ma duceam in directorul unde aveam implementarea temei si checkerul.
- Incarcam modulul cu comanda: module load libraries/cuda-7.5
- Apelam comanda "make" pentru a se crea executabilele.
- Se rula checkerul utilizand comanda: python bench.py

D. Implementare:
================

1. utils.hpp
- Contine doar functie DIE si vectorul de numere prime furnizate de echipa de ASC.

2. gpu_hashtable.hpp
- Contine clasa hashtable-ului, declararea functiilor si diferite define-uri.
- Pe langa metodele clasei hashtable-ului, s-au utilizat urmatoarele functii:
  - 3 functii pentru hash.
  - 3 functii kernel - una pentru insertBatch, una pentru getBatch si una pentru reshape
  - o functie ajutatoare pentru kernelul utilizat la insertBatch
  - o functie ajutatoare pentru kernelul utilizat la reshape

3. gpu_hashtable.cu
- Contine implementarea functiilor.
a. Constructorul - aloca memorie in device pentru hashtable
b. Destructorul - elibereaza memoria alocata pentru hashtable
c. reshape:
- Se retine vechia limita a hashtable-ului si se modifica cu noua limita.
- Se va aloca memorie pentru un hashtable de dimensiunea celui vechi, pentru a retine datele.
Datele din vechiul hashtable vor trebui reintroduse in noul hashtable.
- Se copiaza datele din hastable-ul vechi in hashtable-ul auxiliar.
- Se aloca memorie pentru noul hashtable. Se va aloca memorie in functie de valoare primita
ca parametru la reshape (noua limita).
- Se epeleaza kernelul pentru reshape (insert_pairs_reshape) pentru a reintroduce datele din
vechiul hashtable in noul hashtable.
d. insertBatch:
- Se verifica daca trebuie facut reshape. Daca numarul de elemente introduse si numarul de
elemente ce vor fi introduse este mai mare decat limita, atunci se face reshape.
- Se pun cheile si valorile ce trebuie introduse in device.
- Se calculeaza numarul de blocuri.
- Se apeleaza kernelul pentru inserarea/updatarea valorilor din hashtable (insert_pairs).
e. getBatch:
- Se pun cheile pentru care se cauta valorile pe device.
- Se calculeaza numarul de blocuri si se apeleaza kernelul ce va gasi valorile din hashtable
corespunzatoare cheilor primite (get_value). Aceste valori vor fi puse intr-un vector din
device, iar dupa procesare acesta va fi copiat pe host si returnat.
f. loadFactor:
- Se aduce limita si numarul de elemente inserate de pe device pe host.
- Se returneaza rezultatul impartirii numarului de perechi inserate la limita.
g. Functiile de hash:
- Cele 3 functii de hash inmultesc valoarea primita cu un nr. prim, fac modulo rezultatului cu
un alt numar prim si dupa fac modulo din limita hashtableului.
h. Functiile kernel urmeaza toate aproximativ aceeasi idee:
- Se calculeaza indexul elementului ce trebuie procesat.
- Se verifica daca este in range-ul de elemente corespunzator.
- Se verifica daca cheile sau valorile sunt valide.
- Se calculeaza primul hash si se verifica daca pozitia esta buna sau nu. Daca este buna se
insereaza perechea, updateaza valoarea sau retine valoarea intr-un vector, in functie de caz.
- Daca pozitia nu este buna, se calculeaza al doilea hash. Daca noua pozitie este buna se
insereaza perechea, updateaza valoarea sau retine valoarea intr-un vector, in functie de caz.
- Daca nici aceasta pozitie nu este buna, se calculeaza al treilea hash. Daca noua pozitie este
buna se insereaza perechea, updateaza valoarea sau retine valoarea intr-un vector, in functie
de caz.
- Daca nici pozitia data de al 3-lea hash nu este corecta, ne vom deplasa la "dreapta" pozitie
cu pozitie pana gasim pozitia corespunzatoare si se efectueaza operatiile care trebuie.
i. Functiile ajutatoare:
- Ambele functii verifica daca o pozitie este corespunzatoare si efectueaza operatia
corespunzatoare.
- In check_position_reshape se verifica atomic daca pozitia este libera. Daca este libera
se va pune atomic cheia pe pozitie si apoi se adauga si valoarea.
- In check_position_insert se verifica atomic daca pozitia este libera. Daca este libera
se va pune atomic cheia pe pozitie si apoi se adauga si valoarea si se incrementeaza
numarul total de elemente inserate.  Daca nu este libera se verifica daca este aceeasi cheie.
Daca da, se updateaza valoarea.

E. Exemplu rulare checker:
==========================

$ python bench.py

('HASH_BATCH_INSERT, 100000, inf, 50', ' OK')

('HASH_BATCH_GET, 100000, inf, 50', ' OK')

Test T1 20/20

('HASH_BATCH_INSERT, 2000000, 50, 50', ' OK')

('HASH_BATCH_GET, 2000000, 200, 50', ' OK')

Test T2 20/20

('HASH_BATCH_INSERT, 800000, 40, 50', ' OK')

('HASH_BATCH_INSERT, 800000, 40, 50', ' OK')

('HASH_BATCH_INSERT, 800000, 40, 75', ' OK')

('HASH_BATCH_INSERT, 800000, 40, 50', ' OK')

('HASH_BATCH_INSERT, 800000, 40, 62.5', ' OK')

('HASH_BATCH_GET, 800000, 40, 62.5', ' OK')

('HASH_BATCH_GET, 800000, inf, 62.5', ' OK')

('HASH_BATCH_GET, 800000, inf, 62.5', ' OK')

('HASH_BATCH_GET, 800000, 80, 62.5', ' OK')

('HASH_BATCH_GET, 800000, inf, 62.5', ' OK')

Test T3 10/10


('HASH_BATCH_INSERT, 10000000, 50, 50', ' OK')

('HASH_BATCH_GET, 10000000, 142.857, 50', ' OK')

Test T4 20/20


('HASH_BATCH_INSERT, 2000000, 50, 50', ' OK')

('HASH_BATCH_INSERT, 2000000, 40, 50', ' OK')

('HASH_BATCH_INSERT, 2000000, 40, 75', ' OK')

('HASH_BATCH_INSERT, 2000000, 33.3333, 50', ' OK')

('HASH_BATCH_INSERT, 2000000, 50, 62.5', ' OK')

('HASH_BATCH_GET, 2000000, 100, 62.5', ' OK')

('HASH_BATCH_GET, 2000000, 200, 62.5', ' OK')

('HASH_BATCH_GET, 2000000, 100, 62.5', ' OK')

('HASH_BATCH_GET, 2000000, 100, 62.5', ' OK')

('HASH_BATCH_GET, 2000000, 100, 62.5', ' OK')

Test T5 20/20


TOTAL gpu_hashtable  90/90
