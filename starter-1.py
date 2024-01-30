import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# NumPy Functions
def numpy_average_scores(math_scores, english_scores, science_scores):
    # Calculate average scores in each subject using NumPy
    avg_math = np.mean(math_scores)
    avg_eng = np.mean(english_scores)
    avg_sci = np.mean(science_scores)

    # return all your answers as a tuple
    return avg_math, avg_eng, avg_sci

def numpy_highest_scores(math_scores, english_scores, science_scores):
    # Find the highest scores in each subject using NumPy

    max_math = np.max(math_scores)
    max_eng = np.max(english_scores)
    max_sci = np.max(science_scores)

    # return all your answers as a tuple
    return max_math, max_eng, max_sci

def numpy_highest_science_student(student_ids, science_scores):
    # Find the index of the maximum score in science_scores
    max_sci_index = np.argmax(science_scores)
    
    # Use the index to get the corresponding student ID
    std_id = student_ids[max_sci_index]
    
    # Return std_id as your answer
    return std_id

def numpy_student_4_english_score(student_ids, english_scores):
    # Find what student 4 scored in English using NumPy
    # 1. Get the index number of student 4 (StudentID of 4)
    # 2. Use the index number to filter the english_scores array

    std4_idx = np.where(student_ids == 4)[0][0]
    std4_score = english_scores[std4_idx]

    # Return your answer
    return std4_score

# Pandas Functions
def pandas_create_dataframe(student_ids, math_scores, english_scores, science_scores):
    # Create a dataframe with the given data using Pandas
    df = pd.DataFrame({
        'StudentID': student_ids,
        'Math': math_scores,
        'English': english_scores,
        'Science': science_scores
    })

    # Return df
    return df

def pandas_average_scores(dataframe):
    # Calculate average scores in each subject using Pandas
    avgs = dataframe.mean()
    # Return avgs
    return avgs

def pandas_highest_scores(dataframe):
    # Find the highest scores in each subject using Pandas
    highest_scores = dataframe.max()

    return highest_scores

def pandas_highest_science_student(dataframe):
    # Find the student (Student ID) with the highest score in Science using Pandas
    max_sci_student_id = dataframe.loc[dataframe['Science'].idxmax()]['StudentID']
    return max_sci_student_id

def pandas_student_4_english_score(dataframe):
    # Find what student 4 scored in English using Pandas
    std4_eng_score = dataframe[dataframe['StudentID'] == 4]['English'].values[0]
    return std4_eng_score

def pandas_save_to_csv(df, file):
    # Save the resulting dataframe to a CSV file using Pandas
    df.to_csv(file, index=False)

def pandas_plot_math_distribution(dataframe):
    # Plot a distribution of scores (histogram) in Maths using Pandas
    # use plt.hist function to plot a histogram
    plt.hist(dataframe['Math'])
    plt.show()

# Uncomment the code part
def main():  # call this function to execute the code
    # Example usage for NumPy
    math_scores = np.array([78, 89, 92, 65, 87])
    english_scores = np.array([85, 90, 78, 88, 77])
    science_scores = np.array([90, 92, 88, 75, 80])
    student_ids = np.array([1, 2, 3, 4, 5])

    # Call NumPy functions
    average_scores_np = numpy_average_scores(math_scores, english_scores, science_scores)
    highest_scores_np = numpy_highest_scores(math_scores, english_scores, science_scores)
    highest_science_student_np = numpy_highest_science_student(student_ids, science_scores)
    student_4_english_score_np = numpy_student_4_english_score(student_ids, english_scores)

    print("NumPy Results:")
    print("Average Scores:", average_scores_np)
    print("Highest Scores:", highest_scores_np)
    print("Student with Highest Science Score:", highest_science_student_np)
    print("Student 4 English Score:", student_4_english_score_np)

    # Example usage for Pandas
    df = pandas_create_dataframe(student_ids, math_scores, english_scores, science_scores)

    # Call Pandas functions
    average_scores_pd = pandas_average_scores(df)
    highest_scores_pd = pandas_highest_scores(df)
    highest_science_student_pd = pandas_highest_science_student(df)
    student_4_english_score_pd = pandas_student_4_english_score(df)

    print("\nPandas Results:")
    print("Average Scores:")
    print(average_scores_pd)
    print("\nHighest Scores:")
    print(highest_scores_pd)
    print("\nStudent with Highest Science Score:", highest_science_student_pd)
    print("\nStudent 4 English Score:", student_4_english_score_pd)

    # Save dataframe to CSV
    pandas_save_to_csv(df, 'student_scores_dataframe.csv')

    # Plot distribution of scores in Maths
    pandas_plot_math_distribution(df)

    # Display or print results as needed
main()
