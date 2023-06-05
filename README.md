# ML engineer: Technical Case Study
Thanks for considering joining Gorgias! We believe you could be a great addition to the ML team. 

**Now, we'd love to learn more about how you work.** 

The following case study is aimed at evaluating your ability to solve a business problem, your technical skills, and your ML proficiency. 

# How does it work?

- ✅ **Acknowledge that you read & understood the instructions**
    
    Once you received the case, make sure to **respond to the email** in order to acknowledge that you've properly received & understood all the instructions.
    
- **⏱️ You have 72 hours to work on project [expected effort around 5h]**
    
    Don't forget to send us your project by email before the deadline. We expect:
    
    - an evaluation/backtesting pipeline
    - [optional] an API built on top of your solution
    
    Please upload all your progress on a private GitHub repo. 
    Your first commit should be the project as we send it to you. 
    Make frequent commits to help us follow your journey solving the problem. 
    By the end of the deadline give Victor (victor@gorgias.com) and Firas (firas.jarboui@gorgias.com) reader permissions to your repo. 
    
- 🧪 **You will receive a technical document review assignment**
    
    You need to understand it and prepare a 20 min presentation, we will be expecting the following in the presentation:
    
    - How does this paper relate to the problem at hand?
    - Why we need to explore this research direction?
    - What are the strengths/weaknesses of the suggested approach?
    - Does this approach scale?
    - [optional] some simulations to showcase the main idea.
    
    Feel free to choose another paper, however, make sure that: 
    
    - It relates to the use case
    - published in an A+ venue: NeurIPS, ICLR, ICML, AAAI.
- 📅 **Prepare for the interview**
    - 20 min: presentation of the paper + 10 min QA
    - 40 min: discussion of the case study,
    - 40 min: system design (same project), improvements and caveats
    - 10 min candidate questions

## Project Overview

At Gorgias, we handle merchant customer tickets and provide a software platform that allows the tagging of tickets with the customer contact reason. 
Each client has a **unique set of contact reasons**, making building a flexible and adaptable classification system essential.

Automating the classification of these tickets based on merchant-specific contact reasons can significantly improve efficiency and enhance customer support.

This project aims to develop a machine-learning solution to automate the classification of merchant customer tickets based on their contact reasons. 

Please bear in mind that the final solution would handle about 250K tickets on a daily basis with an expected response time of around 200ms.

## Available Data

The provided dataset is in a Parquet file format, with each line representing a ticket with various attributes. The relevant attributes for classification are as follows:

- account_id: str, Unique merchant ID.
- ticket_id: str, Unique ticket ID.
- raw_body: str, Hashed representation of the received message.
- channel: str, Channel from which the message was received (in this case, emails).
- unix_timestamp: float, Timestamp indicating when the ticket was created.
- contact_reason: str, the contact reason label associated with the message.
- processed_body: str, Hashed representation of the message after processing.
- email_sentence_embeddings: str, JSON string format containing a dictionary of hashed extracted sentences from the email and their embeddings.

To prioritize data privacy, text data has been hashed.
All the embeddings have been generated using a pre-trained transformer (all-MiniLM-L6-v2).

## Project Structure

We recommend that you structure the project as follows:

- `./src/`: Directory containing the source code files.
- `./results/`: Directory to store the performance results.
- `./data/`: Directory containing the dataset in Parquet format.
- `./requirements.txt`: python package requirements for the project.
- `./README.md`: solution documentation and instructions to obtain the results you reached.

## Project output

We will be expecting 

- **a backtesting pipeline**, in the form of a [main.py] file or Jupyter Notebooks, in which we can see how you're evaluating the solution.
- [optional] build **An API on top of your solution** (flask or FastAPI for e.g) that would handle transactional inference. Your endpoint should expect an `account_id` and `email_sentence_embeddings` as input and returns the predicted `contact_reason` as output.

Be thorough in your documentation.

## Project Assessment

We know that working on a project can be time-consuming. However, that's not our goal!
We want to gauge your ability to bring value to Gorgias and see how you can transform your ideas into a clear solution.
For this reason, we would like to provide you with some guidelines on what we are looking for:

- **Neat execution**: We want to see a clean codebase with organised files, comments, and proper code structure.
- **Scientific rigour**: We expect clear ideas and a rigorous evaluation protocol that demonstrates your ability to validate and evaluate your models.
- **Scalable approaches**: We want to see practical solutions that can bring value to our clients and can be scaled effectively as our customer base grows.
- **Creative solutions**: We want you to impress us with your knowledge and propose innovative approaches that demonstrate your understanding of the problem.

You have 72 hours to complete the assessment and send us your solution (by sending the url of a private git repo -in which you give us reader permissions- via email).
We will take the time to review your work and get back to you for a follow-up, during which we will discuss some aspects of your work.

If you have any questions or need assistance throughout the project, please feel free to contact the project team