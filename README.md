# Document Classifier

This is a basic Java-based machine learning project designed to classify technical documents using a preprocessed `.arff` dataset format (used by Weka). The project was developed in Eclipse and includes source code, compiled classes, and sample data.

---

## Project Structure

```
Document-Classifier/
├── .classpath
├── .project
├── arff.arff
├── bin/
│   └── classifier/
│       └── DocumentClassifier.class
├── src/
│   └── classifier/
│       └── DocumentClassifier.java
└── .settings/
    └── org.eclipse.jdt.core.prefs
```

- `src/classifier/DocumentClassifier.java` – Main source file. Loads `.arff` file and applies classification using Weka.
- `arff.arff` – Dataset in Weka's ARFF format (Attribute-Relation File Format).
- `bin/` – Eclipse auto-generated compiled `.class` files.
- `.classpath`, `.project`, `.settings/` – Eclipse metadata.

---

## Prerequisites

- Java 8 or higher
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/) (Weka JAR file, e.g., `weka-stable.jar`)
- Optional: Eclipse IDE (for easier navigation)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/srishsriram/Document-Classifier.git
cd Document-Classifier
```

### 2. Add Weka to Classpath

Download `weka-stable.jar` and place it in the project root or a `lib/` folder.

If compiling manually:

```bash
javac -cp weka-stable.jar src/classifier/DocumentClassifier.java -d bin
```

If running manually:

```bash
java -cp "bin:weka-stable.jar" classifier.DocumentClassifier
```

Use `;` instead of `:` on Windows for classpath separation.

---

## Sample `Main` Method Behavior

This is the expected flow (based on `DocumentClassifier.java`):

1. Load dataset from `arff.arff`
2. Initialize classifier (e.g., NaiveBayes or J48)
3. Train classifier on dataset
4. Evaluate model using cross-validation
5. Output summary statistics

---

## Sample Code Snippet

```java
Instances data = new Instances(new BufferedReader(new FileReader("arff.arff")));
data.setClassIndex(data.numAttributes() - 1);

Classifier classifier = new J48(); // Can switch to NaiveBayes, etc.
classifier.buildClassifier(data);

Evaluation eval = new Evaluation(data);
eval.crossValidateModel(classifier, data, 10, new Random(1));

System.out.println(eval.toSummaryString());
```

---

## ARFF File

The `arff.arff` file must define:

- A set of attributes (numeric, nominal, string)
- A target attribute (last column) to classify
- A dataset section (`@data`) with matching rows

Example (partial):

```
@RELATION docs
@ATTRIBUTE length NUMERIC
@ATTRIBUTE has_code {yes,no}
@ATTRIBUTE class {technical,nontechnical}
@DATA
100,yes,technical
```

## License

Personal or academic use only.

## Author

Srish Sriram – [github.com/srishsriram](https://github.com/srishsriram)
````
