/* Copyright [2019] Gavan Adrian-George, 334CA */
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"
#include "utils.hpp"


/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

	// Retine codul intors de unele functii cuda, pentru a verifica erori
	cudaError_t ret_code;

	// Se aloca memorie pentru vectorii si variabilele din VRAM

	ret_code = cudaMalloc((void **)&limit_device, size * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc1 Init");
	ret_code = cudaMemcpy(limit_device, &size, sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy1 Init");

	ret_code = cudaMalloc((void **)&htab_keys_device, size * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc2 Init");
	ret_code = cudaMemset(htab_keys_device, 0, size * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset1 Init");

	ret_code = cudaMalloc((void **)&htab_values_device, size * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc3 Init");
	ret_code = cudaMemset(htab_values_device, 0, size * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset2 Init");

	ret_code = cudaMalloc((void **)&nr_inserted_device, sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc4 Init");
	ret_code = cudaMemset(nr_inserted_device, 0, sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset3 Init");

}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {

	// Se elibereaza memoria din VRAM

	cudaFree(htab_keys_device);
	cudaFree(htab_values_device);
	cudaFree(nr_inserted_device);
	cudaFree(limit_device);

}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	// Retine codul intors de unele functii cuda, pentru a verifica erori
	cudaError_t ret_code;
	// Variabila locala pentru a luat limita de pe VRAM
	int limit = 0;
	// Se retine limita de dinainte de reshape
	int old_limit = 0;
	// Vechiul vector de chei din VRAM
	int* htab_keys_device_old = NULL;
	// Vechiul vector de valori din VRAM
	int* htab_values_device_old = NULL;
	// Numarul de blocuri
	size_t blocks_no = 0;

	// Se copiaza limita din device pe host
	ret_code = cudaMemcpy(&limit, limit_device, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy1 Reshape");

	// Se modifica limita
	old_limit = limit;
	limit = numBucketsReshape;

	// Se pune limita modificata pe device
	ret_code = cudaMemcpy(limit_device, &limit, sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy2 Reshape");

	// Se aloca memorie pentru a se retine vechiul hashtable
	ret_code = cudaMalloc((void **)&htab_keys_device_old, old_limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc1 Reshape");
	ret_code = cudaMalloc((void **)&htab_values_device_old, old_limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc2 Reshape");

	// Se copiaza hashtable-ul in vectorii auxiliari pentru
	// a se retine ce valori erau in hashtable
	ret_code = cudaMemcpy(htab_keys_device_old, htab_keys_device,
			old_limit * sizeof(int), cudaMemcpyDeviceToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy3 Reshape");
	ret_code = cudaMemcpy(htab_values_device_old, htab_values_device,
			old_limit * sizeof(int), cudaMemcpyDeviceToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy4 Reshape");

	// Se elibereaza memoria hashtable-ului
	cudaFree(htab_keys_device);
	cudaFree(htab_values_device);

	// Se aloca noul hashtable cu noua limita
	ret_code = cudaMalloc((void **)&htab_keys_device, limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc3 Reshape");
	ret_code = cudaMemset(htab_keys_device, 0, limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset1 Reshape");

	ret_code = cudaMalloc((void **)&htab_values_device, limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc4 Reshape");
	ret_code = cudaMemset(htab_values_device, 0, limit * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset1 Reshape");

	// Se calculeaza numarul de blocuri
	blocks_no = old_limit / BLOCK_SIZE;
	if (old_limit % BLOCK_SIZE) {
		blocks_no++;
	}

	// Se insereaza elementele care deja erau in hashtable in
	// noul hashtable
	insert_pairs_reshape<<<blocks_no, BLOCK_SIZE>>>(htab_keys_device_old,
		htab_values_device_old, limit_device, old_limit,
		htab_keys_device, htab_values_device);
	cudaDeviceSynchronize();

	// Se elibereaza hashable-ul auxiliar
	cudaFree(htab_keys_device_old);
	cudaFree(htab_values_device_old);
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	// Retine codul intors de unele functii cuda, pentru a verifica erori
	cudaError_t ret_code;
	// Variabila in care se copiaza limita din device pe host
	int limit = 0;
	// Variabila in care se copiaza nr. de elemente inserate
	// din device pe host
	int nr_inserted = 0;
	// Vectori in care se vor pune cheile si valorile de pe host
	// pe device
	int* values_device = NULL;
	int* keys_device = NULL;
	// Numarul de blocuri
	size_t blocks_no = 0;

	// Se copiaza pe host limita si nr. de elemente inserate
	ret_code = cudaMemcpy(&limit, limit_device, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy1 Insert");
	ret_code = cudaMemcpy(&nr_inserted, nr_inserted_device, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy2 Insert");

	// Daca numarul de elemente inserate deja + numarul
	// elementelor ce trebuie introduse depasesc limita maxima,
	// atunci se apeleaza functia reshape.
	// Se utilizeaza aceasta formula la reshape pentru ca in cel mai
	// rau caz vom avea 50% load factor si maxim 100% load factor.
	if (nr_inserted + numKeys >= limit) {
		reshape((nr_inserted + numKeys) * RESHAPE_FACTOR);
	}

	// Se aloca memorie pentru a pune cheile si valorile primite in device
	ret_code = cudaMalloc((void **)&values_device, numKeys * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc1 Insert");
	ret_code = cudaMalloc((void **)&keys_device, numKeys * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc2 Insert");

	// Se calculeaza numarul de blocuri
	blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) {
		blocks_no++;
	}

	// Se copiaza cheile si valorile in device
	ret_code = cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy3 Insert");
	ret_code = cudaMemcpy(values_device, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy4 Insert");

	// Se insereaza cheile si valorile in hashtable (sau se updateaza
	// valorile daca cheia deja exista)
	insert_pairs<<<blocks_no, BLOCK_SIZE>>>(keys_device, values_device,
		limit_device, numKeys, htab_keys_device,
		htab_values_device, nr_inserted_device);
	cudaDeviceSynchronize();

	// Se elibereaza memoria
	cudaFree(values_device);
	cudaFree(keys_device);

	// Conform cerintei nu s-a tratat cazul in care nu se pot insera
	// elementele, asa ca se va returna mereu true
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	// Retine codul intors de unele functii cuda, pentru a verifica erori
	cudaError_t ret_code;
	// Vectorul de pe device unde se vor pune valorile ce
	// trebuie returnate
	int *ret_values_device = NULL;
	// Vectorul de pe device in care se pun cheile primite
	// ca parametru
	int *keys_device = NULL;
	// Numarul de blocuri
	size_t blocks_no = 0;
	// Vectorul cu valori ce va fi returnat
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	DIE(ret_values == NULL, "Malloc Get");

	// Se aloca memorie pentru vectorii de pe device
	ret_code = cudaMalloc((void **)&ret_values_device, numKeys * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc1 Get");
	ret_code = cudaMalloc((void **)&keys_device, numKeys * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMalloc1 Get");

	// Se copiaza cheile de pe host pe device
	ret_code = cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret_code != cudaSuccess, "cudaMemcpy1 Get");

	// Se pune 0 in vectorul ce trebuie returnat de pe device
	// astfel incat daca nu se va gasi valoarea pentru o cheie
	// atunci pe pozitia corespunzatoare va fi 0 (in implementare
	// 0 nu este o cheie sau o valoare valida)
	ret_code = cudaMemset(ret_values_device, 0, numKeys * sizeof(int));
	DIE(ret_code != cudaSuccess, "cudaMemset1 Get");

	// Se calculeaza numarul de blocuri
	blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) {
		blocks_no++;
	}

	// Se determina vectorul cu valorile corespunzatoare cheilor
	// primite ca parametru
	get_values<<<blocks_no, BLOCK_SIZE>>>(ret_values_device, keys_device,
		limit_device, numKeys, htab_keys_device, htab_values_device);
	cudaDeviceSynchronize();

	// Se copiaza rezultatele de pe device pe host
	ret_code = cudaMemcpy(ret_values, ret_values_device, numKeys * sizeof(int),
						cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy2 Get");

	// Se elibereaza memoria
	cudaFree(ret_values_device);
	cudaFree(keys_device);

	// Se returneaza rezultatul
	return ret_values;

}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {

	// Retine codul intors de unele functii cuda, pentru a verifica erori
	cudaError_t ret_code;
	int nr_inserted = 0;
	int limit = 0;

	// Se aduc datele din device pe host
	cudaMemcpy(&limit, limit_device, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy1 LoadFactor");
	cudaMemcpy(&nr_inserted, nr_inserted_device, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(ret_code != cudaSuccess, "cudaMemcpy2 LoadFactor");

	// Se returneaza load factor-ul
	return (float)nr_inserted / (float)limit;

}


/********** Implementarea kernel-urilor **********/


__global__ void insert_pairs_reshape(int *keys_device,
			int *values_device, int *limit,
			int nr_keys, int *hashtable_keys,
			int *hashtable_values) {

	// Se afla indexul elementului care trebuie procesat
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	// Variabila utilizata pentru retinerea ulimei valori
	// returnate de o functie de hash
	int hash_code = 0;
	// Variabila utilizata pentru pozitia in hashtable
	int position = 0;

	// Daca indexul este in afara vectorului de chei
	// se termina executia kernel-ului
	if(index >= nr_keys) {
		return;
	}

	// Daca cheia este 0 se termina executia kernel-ului
	if(keys_device[index] == KEY_INVALID) {
		return;
	}

	// Se calculeaza prima valoare de hash utilizand prima functie
	hash_code = hash_function1(keys_device[index], *limit);

	// Se verifica daca pozitia este valida.
	// Daca da, se va insera cheia si valoarea
	// si se va returna true. Altfel sa va returna false.
	if(check_position_reshape(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device)) {

		return;
	}

	// Daca primul hash nu a intors o pozitie corespunzatoare, se
	// calculeaza un alt hash cu a doua functie.
	hash_code = hash_function2(keys_device[index], *limit);

	// Se verifica daca pozitia este valida.
	// Daca da, se va insera cheia si valoarea
	// si se va returna true. Altfel sa va returna false.
	if(check_position_reshape(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device)) {

		return;
	}

	// Daca nici al doilea hash nu a intors o pozitie corespunzatoare,
	// se calculeaza un alt hash cu a treia functie.
	hash_code = hash_function3(keys_device[index], *limit);

	// Se verifica daca pozitia este valida.
	// Daca da, se va insera cheia si valoarea
	// si se va returna true. Altfel sa va returna false.
	if(check_position_reshape(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device)) {

		return;
	}

	// Daca nici a treia functie nu intoarce o valoare corespunzatoare,
	// Ne vom deplasa la dreapta pana gasim o pozitie goala sau aceeasi
	// cheie.
	// Se calculeaza prima pozitie de la dreapta
	position = (hash_code + 1) % (*limit);

	// Se verifica pozitiile pana cand se ajunge in pozitia initiala.
	// Nu ar trebui sa se intample acest caz pentru ca toate elementele
	// trebuie sa aiba loc in hashtable
	while(position != hash_code) {
		// Se verifica daca pozitia este valida.
		// Daca da, se va insera cheia si valoarea
		// si se va returna true. Altfel sa va returna false.
		if(check_position_reshape(position, index, hashtable_keys,
				hashtable_values, keys_device, values_device)) {

			return;
		}

		// Se trece la urmatoarea pozitie
		position = (position + 1) % (*limit);
	}

}

__global__ void insert_pairs(int *keys_device, int *values_device,
			int* limit, int nr_keys, int* hashtable_keys,
			int* hashtable_values, int* nr_inserted) {

	// Se afla indexul elementului care trebuie procesat
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	// Variabila utilizata pentru retinerea ulimei valori
	// returnate de o functie de hash
	int hash_code = 0;
	// Variabila utilizata pentru pozitia in hashtable
	int position = 0;

	// Daca indexul este in afara vectorului de chei
	// se termina executia kernel-ului
	if(index >= nr_keys) {
		return;
	}

	// Daca cheia sau valoarea este invalida se termina
	// executia kernel-ului
	if(keys_device[index] <= KEY_INVALID || values_device <= KEY_INVALID) {
		return;
	}

	// Se calculeaza prima valoare de hash utilizand prima functie
	hash_code = hash_function1(keys_device[index], *limit);

	// Se verifica daca pozitia indicata de hash este valida.
	// Daca da, se va insera cheia si valoarea (sau se va updata
	// valoarea) si se va returna true. Altfel sa va returna false.
	if(check_position_insert(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device, nr_inserted)) {

		return;
	}

	// Daca primul hash nu a intors o pozitie corespunzatoare, se
	// calculeaza un alt hash cu a doua functie.
	hash_code = hash_function2(keys_device[index], *limit);

	// Se verifica daca pozitia indicata de hash este valida.
	// Daca da, se va insera cheia si valoarea (sau se va updata
	// valoarea) si se va returna true. Altfel sa va returna false.
	if(check_position_insert(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device, nr_inserted)) {

		return;
	}

	// Daca nici al doilea hash nu a intors o pozitie corespunzatoare,
	// se calculeaza un alt hash cu a treia functie.
	hash_code = hash_function3(keys_device[index], *limit);

	// Se verifica daca pozitia indicata de hash este valida.
	// Daca da, se va insera cheia si valoarea (sau se va updata
	// valoarea) si se va returna true. Altfel sa va returna false.
	if(check_position_insert(hash_code, index, hashtable_keys,
			hashtable_values, keys_device, values_device, nr_inserted)) {

		return;
	}

	// Daca nici a treia functie nu intoarce o valoare corespunzatoare,
	// Ne vom deplasa la dreapta pana gasim o pozitie goala sau aceeasi
	// cheie.
	// Se calculeaza prima pozitie de la dreapta
	position = (hash_code + 1) % (*limit);

	// Se verifica pozitiile pana cand se ajunge in pozitia initiala.
	// Nu ar trebui sa se intample acest caz pentru ca toate elementele
	// trebuie sa aiba loc in hashtable
	while(position != hash_code) {

		// Se verifica daca pozitia este valida.
		// Daca da, se va insera cheia si valoarea (sau se va updata
		// valoarea) si se va returna true. Altfel sa va returna false.
		if(check_position_insert(position, index, hashtable_keys,
				hashtable_values, keys_device, values_device, nr_inserted)) {

			return;
		}

		// Se trece la urmatoarea pozitie
		position = (position + 1) % (*limit);
	}

}

__global__ void get_values(int *result, int *searched_keys, int* limit,
			int nr_keys, int* hashtable_keys, int* hashtable_values) {

	// Se afla indexul elementului care trebuie procesat
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	// Variabila utilizata pentru retinerea ulimei valori
	// returnate de o functie de hash
	int hash_code = 0;
	// Variabila utilizata pentru pozitia in hashtable
	int position = 0;

	// Daca indexul este in afara vectorului de chei
	// se termina executia kernel-ului
	if (index >= nr_keys) {
		return;
	}

	// Se calculeaza prima valoare de hash utilizand prima functie
	hash_code = hash_function1(searched_keys[index], *limit);

	// Daca cheia este la pozitia indicata de hash =>
	// se retine valoarea de pe pozitia respectiva
	if(hashtable_keys[hash_code] == searched_keys[index]) {
		result[index] = hashtable_values[hash_code];
		return;
	}

	// Daca primul hash nu a intors o pozitie corespunzatoare, se
	// calculeaza un alt hash cu a doua functie.
	hash_code = hash_function2(searched_keys[index], *limit);

	// Daca cheia este la pozitia indicata de hash =>
	// se retine valoarea de pe pozitia respectiva
	if(hashtable_keys[hash_code] == searched_keys[index]) {
		result[index] = hashtable_values[hash_code];
		return;
	}

	// Daca nici al doilea hash nu a intors o pozitie corespunzatoare,
	// se calculeaza un alt hash cu a treia functie.
	hash_code = hash_function3(searched_keys[index], *limit);

	// Daca cheia este la pozitia indicata de hash =>
	// se retine valoarea de pe pozitia respectiva
	if(hashtable_keys[hash_code] == searched_keys[index]) {
		result[index] = hashtable_values[hash_code];
		return;
	}

	// Daca nici a treia functie nu intoarce o valoare corespunzatoare,
	// Ne vom deplasa la dreapta pana gasim o pozitie goala sau aceeasi
	// cheie.
	// Se calculeaza prima pozitie de la dreapta
	position = (hash_code + 1) % (*limit);

	// Se verifica pozitiile pana cand se ajunge in pozitia initiala.
	// Nu ar trebui sa se intample acest caz pentru ca toate elementele
	// trebuie sa aiba loc in hashtable
	while(position != hash_code){
		// Daca cheia este la pozitia indicata de hash =>
		// se retine valoarea de pe pozitia respectiva
		if (hashtable_keys[position] == searched_keys[index]) {
			result[index] = hashtable_values[position];
			return;
		}

		// Se trece la urmatoarea pozitie
		position = (position + 1) % (*limit);
	}

}


/********** Implementarea functiilor de hash **********/


__device__ int hash_function1(int data, int limit) {
	return ((long)abs(data) * HASH1_FACT1) % HASH1_FACT2 % limit;
}

__device__ int hash_function2(int data, int limit) {
	return ((long)abs(data) * HASH2_FACT1) % HASH2_FACT2 % limit;
}

__device__ int hash_function3(int data, int limit) {
	return ((long)abs(data) * HASH3_FACT1) % HASH3_FACT2 % limit;
}


/********** Implementare functii auxiliare **********/


__device__ bool check_position_reshape(int position, int index,
			int *hashtable_keys, int *hashtable_values,
			int *keys_device, int *values_device) {

	// Daca position indica o pozitie goala => se adauga
	// cheia si valoarea.
	// Cheia este adaugata atomic de atomicCAS.
	if(atomicCAS(&hashtable_keys[position], EMPTY_POS, keys_device[index]) == EMPTY_POS) {
		hashtable_values[position] = values_device[index];
		return true;
	}

	return false;
}

__device__ bool check_position_insert(int position, int index,
				int *hashtable_keys, int *hashtable_values,
				int *keys_device, int *values_device, int *nr_inserted) {

	// Daca pozitia indica o pozitie goala => se adauga cheia
	// si valoarea si se incrementeaza nr. de elemente inserate
	// Cheia este adaugata atomic de atomicCAS.
	if((atomicCAS(&hashtable_keys[position], EMPTY_POS, keys_device[index]) == EMPTY_POS)) {
		hashtable_values[position] = values_device[index];
		atomicAdd(nr_inserted, 1);
		return true;
	}

	// Daca este exact cheia ce trebuie updatata =>
	// se adauga cheia si valoarea
	if (hashtable_keys[position] == keys_device[index]) {
		hashtable_values[position] = values_device[index];
		return true;
	}

	return false;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
