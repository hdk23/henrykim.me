/* 
 * bagtest.c - test program for CS50 bag module
 *
 * usage: read lines from stdin
 *
 * CS50, Summer 2021
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bag.h"
#include "readlinep.h"

static void nameprint(FILE *fp, void *item);
static void namedelete(void *item);
static void itemcount(void *arg, void *item);

/* **************************************** */
int main() 
{
  bag_t *bag;
  int namecount = 0;

  // create a bag
  bag = bag_new();
  if (bag == NULL) {
    fprintf(stderr, "bag_new failed\n");
    return 1;
  }

  printf("Count (should be zero): ");
  bag_iterate(bag, &namecount, itemcount);
  printf("%d\n", namecount);  

  printf("testing bag_insert...\n");
  // read lines from stdin
  namecount = 0;
  while (!feof(stdin)) {
    char *name = readlinep();
    if (name != NULL) {
      bag_insert(bag, name);
      namecount++;
    }
  }

  printf("Count (should be %d): ", namecount);
  namecount = 0;
  bag_iterate(bag, &namecount, itemcount);
  printf("%d\n", namecount);  

  printf("test with null bag, good item...\n");
  bag_insert(NULL, "Dartmouth");
  printf("test with null item...\n");
  bag_insert(bag, NULL); 
  printf("test with null bag, null item...\n");
  bag_insert(NULL, NULL);

  printf("The bag:\n");
  bag_print(bag, stdout, nameprint);
  printf("\n");

  printf("delete the bag...\n");
  bag_delete(bag, namedelete);

  return 0;
}


/* count the non-null items in the bag.
 * note here we don't care what kind of item is in bag.
 */
static void itemcount(void *arg, void *item)
{
  int *nitems = arg;

  if (nitems != NULL && item != NULL)
    (*nitems)++;
}

// print a name, in quotes.
void nameprint(FILE *fp, void *item)
{
  char *name = item; 
  if (name == NULL) {
    fprintf(fp, "(null)");
  }
  else {
    fprintf(fp, "\"%s\"", name); 
  }
}

// delete a name 
void namedelete(void *item)
{
  if (item != NULL) {
    free(item);   
  }
}
