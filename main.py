
if __name__ == '__main__':
    # run part A
    print('...PART A...\n\n')
    A = open('analyses.py')
    run_file = A.read()
    exec(run_file)

    # run part B
    print('...PART B...\n\n')
    B = open('baselines.py')
    run_file = B.read()
    exec(run_file)

    # run part C
    print('...PART C...\n\n')

    C1 = open('build_vocab.py')
    run_file = C1.read()
    exec(run_file)

    C2 = open('train.py')
    run_file = C2.read()
    exec(run_file)

    C3 = open('evaluate.py')
    run_file = C3.read()
    exec(run_file)

    C4 = open('detailed_evaluation.py')
    run_file = C4.read()
    exec(run_file)
