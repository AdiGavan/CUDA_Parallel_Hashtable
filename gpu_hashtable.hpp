/* Copyright [2019] Gavan Adrian-George, 334CA */
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID	0
#define BLOCK_SIZE 256
#define RESHAPE_FACTOR 2
#define EMPTY_POS 0

#define HASH1_FACT1 129607llu
#define HASH1_FACT2 26339969llu
#define HASH2_FACT1 518509llu
#define HASH2_FACT2 66372617llu
#define HASH3_FACT1 2074129llu
#define HASH3_FACT2 132745199llu

/* Prima functie de hash */
__device__ int hash_function1(int data, int limit);
/* A doua functie de hash */
__device__ int hash_function2(int data, int limit);
/* A treia functie de hash */
__device__ int hash_function3(int data, int limit);

/**
 * Functie folosita in reshape care verifica daca pe pozitia
 * curenta se gaseste o pozitie libera.
 * Daca pozitia este valida, se introduce cheia si valoarea.
 * Se returneaza true daca pozitia a fost valida, false altfel
 */
__device__ bool check_position_reshape(int position, int index,
			int *hashtable_keys, int *hashtable_values,
			int *keys_device, int *values_device);

/**
 * Functie folosita in insert care verifica daca pe pozitia
 * curenta se gaseste o pozitie libera sau chiar cheia ce
 * trebuie introdusa (update).
 * Daca pozitia este valida, se introduce cheia si valoarea.
 * Se returneaza true daca pozitia a fost valida, false altfel
 */
__device__ bool check_position_insert(int position, int index,
			int *hashtable_keys, int *hashtable_values,
			int *keys_device, int *values_device, int *nr_inserted);

/**
 * Kernel utilizat cand se face un reshape al hashtable-ului,
 * pentru a reintroduce perechile deja existente in hashtable
 */
__global__ void insert_pairs_reshape(int *keys_device,
			int *values_device, int *limit,
			int nr_keys, int* hashtable_keys,
			int* hashtable_values);

/**
 * Kernel utilizat pentru a insera perechi sau a updata valori
 */
__global__ void insert_pairs(int *keys_device, int *values_device, int *limit,
			int nr_keys, int *hashtable_keys,
			int *hashtable_values, int *inserted);

/**
 * Kernel utilizat pentru returnarea valorilor unor anumite chei
 */
__global__ void get_values(int *result, int *searched_keys, int *limit,
			int nr_keys, int *hashtable_keys, int *hashtable_values);

/**
 * GPU HashTable
 */
class GpuHashTable
{
	private:
		// Vector utilizat pentru a stoca cheila in VRAM
		int* htab_keys_device;
		// Vector utilizat pentru a stoca valorile in VRAM
		int* htab_values_device;
		// Variabila utilizata pentru a tine dimensiunea
		// hashtable-ului in VRAM
		int* limit_device;
		// Variabila utilizata pentru a tine numarul de perechi
		// inserate in hashtable in VRAM
		int* nr_inserted_device;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		float loadFactor();
		void occupancy();
		void print(string info);

		~GpuHashTable();
};

#endif
