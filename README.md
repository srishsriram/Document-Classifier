# Document-Classifier

**Goal**: Train and evaluate classic ML text classifiers (e.g., Naive Bayes, SVM) using Weka and libsvm on ARFF-formatted data.

---

## Features
- ARFF-driven dataset input (`arff.arff`).
- Uses Weka filters and classifiers, plus libsvm.
- CLI-friendly main program (`Categorization.java`).
- Eclipse project files included.

---

## Project Structure
```
Document-Classifier-master/
├── .classpath
├── .project
├── .settings/
│   └── org.eclipse.jdt.core.prefs
├── README.md
├── arff.arff
├── bin/
│   └── Categorization.class
└── src/
    └── Categorization.java
```

---

## Dependencies
- Java 8+ (OpenJDK or Oracle JDK).
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/) JARs on classpath.
- [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) JAR on classpath.

> Ensure `weka.jar` and `libsvm.jar` paths are available locally.

---

## Build

### With `javac`
```bash
# Paths to your jars
WEKA_JAR=/path/to/weka.jar
LIBSVM_JAR=/path/to/libsvm.jar

# From project root
javac -cp "$WEKA_JAR:$LIBSVM_JAR:." -d bin src/Categorization.java
```

### With Eclipse
- Import → Existing Projects into Workspace → select `Document-Classifier-master`.
- Add Weka and libsvm JARs to Project Properties → Java Build Path → Libraries.

---

## Run
```bash
# Example: classify using the bundled ARFF
java -cp "bin:$WEKA_JAR:$LIBSVM_JAR:." Categorization \
  --train arff.arff \
  --test  arff.arff \
  --algo  naive_bayes
```

> Flags may vary. If no CLI parsing exists, edit constants in `Categorization.java` or modify the code to accept arguments.

---

## Data Format (ARFF)
The sample `arff.arff` defines free-text and a nominal class. Example excerpt:
```arff
@relation test

@attribute text string
@attribute @class@ {faq,forum,kb,manual,news}

@data
"How do I perform a live vMotion of a Delphix Engine?", faq
"Does the scheduler support cron expressions?", kb
```
- `text`: free-form string feature.
- `@class@`: target label; update the label set for your use case.

---

## Core Class

### `src/Categorization.java`
- Imports Weka core, filters, and classifiers; imports libsvm.
- Contains a `public static void main(...)` entrypoint.
- Reads ARFF paths, builds features, trains, and evaluates a model.
- Example detected constructs:
  - Uses Weka classifiers such as `NaiveBayes`.
  - Applies Weka filters (`weka.filters.*`) for preprocessing.
  - Integrates `libsvm` for SVM models.

Excerpt:
```java
import weka.core.*;
import libsvm.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.*;
// ...
public class Categorization {
    public static void main(String[] args) throws Exception {
        // load ARFF, build classifier, evaluate
        // e.g., DataSource source = new DataSource(trainPath);
        // Instances train = source.getDataSet();
        // train.setClassIndex(train.numAttributes() - 1);
        // Classifier cls = new NaiveBayes();
        // cls.buildClassifier(train);
        // ...
    }
}
```

---

## Typical Pipeline
1. Load ARFF train and test sets.
2. Set class index to last attribute.
3. Optionally apply filters (e.g., `StringToWordVector` for text).
4. Train a classifier (Naive Bayes, SVM via libsvm).
5. Evaluate on test set (accuracy, precision/recall, confusion matrix).

---

## Extending
- Replace Naive Bayes with other Weka classifiers (`J48`, `Logistic`, `SMO`).
- Add `StringToWordVector` with TF-IDF, n-grams, stopwords.
- Save and reload models with Weka Serialization.
- Add CLI options with a parser (e.g., args4j or simple custom parsing).

---

## Notes and Gaps
- No build tool files (Maven/Gradle). Add one for reproducibility.
- No explicit CLI help or flags documented in code. Consider adding.
- No tests included. Add unit tests and small ARFF fixtures.
- Ensure jar versions of Weka and libsvm match the imports used.

---

## Quick Start
```bash
# 1) Prepare data
cp your_dataset.arff arff.arff

# 2) Compile
javac -cp "$WEKA_JAR:$LIBSVM_JAR:." -d bin src/Categorization.java

# 3) Run
java -cp "bin:$WEKA_JAR:$LIBSVM_JAR:." Categorization
```
