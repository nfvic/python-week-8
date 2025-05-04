# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Apply seaborn styling for better visuals
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    try:
        # Task 1: Load and Explore Dataset
        print("=== Task 1: Loading and Exploring Data ===")
        
        # Load dataset from sklearn and convert to DataFrame
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]

        # Save as CSV to simulate real-world usage
        df.to_csv('iris.csv', index=False)
        df = pd.read_csv('iris.csv')

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

        # Clean missing values (Iris has none, but shown for demo)
        df.dropna(inplace=True)

        # Task 2: Basic Data Analysis
        print("\n\n=== Task 2: Basic Data Analysis ===")
        print("\nDescriptive Statistics:")
        print(df.describe())

        print("\nMean values grouped by species:")
        print(df.groupby('species').mean())

        # Task 3: Data Visualization
        print("\n\n=== Task 3: Data Visualization ===")

        # 1. Line Chart
        plt.figure()
        df['sepal length (cm)'].plot()
        plt.title('Sepal Length Over Samples')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        plt.show()

        # 2. Bar Chart
        plt.figure()
        df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['red', 'green', 'blue'])
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        plt.xticks(rotation=0)
        plt.show()

        # 3. Histogram
        plt.figure()
        df['sepal width (cm)'].hist(bins=15, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()

        # 4. Scatter Plot
        plt.figure()
        colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
        plt.scatter(df['sepal length (cm)'], df['petal length (cm)'],
                    c=df['species'].map(colors), alpha=0.6)
        plt.title('Sepal vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
                   for color in colors.values()]
        plt.legend(handles, colors.keys(), title='Species')
        plt.show()

        # Summary
        print("\n=== Key Findings ===")
        print("• Setosa species has the smallest petal measurements.")
        print("• Sepal width appears normally distributed.")
        print("• Sepal and petal lengths are positively correlated.")
        print("• Virginica shows the highest average values across features.")

    except FileNotFoundError:
        print("Error: iris.csv not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
