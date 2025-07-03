def debug_print_examples(train_dataset, n=5):
    print("\n===== DEBUG: Exemples du train_dataset après mapping =====\n")
    for i in range(min(n, len(train_dataset))):
        print(f"Exemple {i} :\n{train_dataset[i]['text']}")
        print("-" * 80) 