"""Build the character-name pool from names REAL AITA posters actually used.

The pool is not invented. Candidate names are extracted from the human corpus
and kept only if they appear on a curated gendered person-name list, and each
one carries the frequency it had in the human class. Drawing from that pool
reproduces the human name distribution BY CONSTRUCTION -- including its mild
recurrence, which is the point.

Flattening to a unique name per document would overshoot into "more
name-diverse than any real population", which is phase 1's v4 failure in a new
place: absence and excess are both detectable, only the rate is neutral.

The curated list exists because the extractor in aita_08_diversity.py is a
capitalisation heuristic, not a person-name detector. Run raw over the human
corpus it returns Well 42, Now 23, Netflix 22, Hey 18, Just 17 among its top
names. A draw built from that would hand models "Netflix" as a character.

    python3 phase3/scripts/aita_11_name_pool.py
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/phase3/scripts")
from aita_08_diversity import names_in  # noqa: E402

FEMALE = """
Abigail Ada Adriana Aimee Alexa Alexandra Alexis Alice Alicia Alison Allison Amanda Amber Amelia
Amy Andrea Angela Angie Anita Ann Anna Anne Annie April Ashley Audrey Autumn Ava Barbara Beatrice
Becca Becky Bella Beth Bethany Betty Beverly Bianca Bonnie Brenda Bridget Brittany Brooke Caitlin
Cameron Camila Candace Cara Carla Carmen Carol Caroline Carrie Cassandra Cassie Catherine Cathy
Charlotte Chelsea Cheryl Chloe Christina Christine Cindy Claire Clara Colleen Connie Courtney
Crystal Cynthia Daisy Dana Danielle Daphne Darlene Dawn Deborah Debra Delia Denise Diana Diane
Dianne Donna Dora Doris Dorothy Eileen Elaine Eleanor Elena Elise Elizabeth Ella Ellen Emily Emma
Erica Erin Esther Eva Evelyn Faith Felicia Fiona Frances Gabriela Gail Gemma Georgia Gina Ginny
Gloria Grace Gwen Hannah Harriet Hayley Heather Heidi Helen Holly Hope Imogen Irene Iris Isabel
Isabella Ivy Jacqueline Jade Jamie Jane Janet Janice Jasmine Jean Jeanne Jenna Jennifer Jenny
Jessica Jill Joan Joanna Jocelyn Jodie Joy Joyce Judith Judy Julia Julie June Karen Kate Katherine
Kathleen Kathy Katie Katrina Kayla Kelly Kelsey Kendra Kerry Kim Kimberly Kirsten Kristen Kristin
Lacey Larissa Laura Lauren Leah Leanne Lena Leslie Lila Lilian Lily Linda Lindsay Lisa Liz Lois
Lorraine Louise Lucy Luna Lydia Lynn Mabel Maddie Madeleine Madison Maggie Mandy Marcia Margaret
Maria Mariah Marie Marilyn Marion Marisa Marjorie Martha Mary Maureen Maya Meagan Megan Melanie
Melissa Meredith Mia Michelle Mila Mildred Miranda Miriam Molly Monica Nadia Nancy Naomi Natalie
Natasha Nicole Nina Nora Norma Olivia Paige Pam Pamela Patricia Paula Pauline Pearl Peggy Penny
Phoebe Phyllis Priya Rachel Rebecca Regina Renee Rhonda Rita Roberta Robin Rosa Rose Rosemary Ruby
Ruth Sabrina Sadie Sally Samantha Sandra Sara Sarah Savannah Selena Shannon Sharon Sheila Shelly
Sherry Shirley Sofia Sonia Sophia Sophie Stacey Stella Stephanie Sue Susan Suzanne Sylvia Tabitha
Tamara Tammy Tanya Tara Teresa Tessa Thelma Theresa Tiffany Tina Tonya Tracy Trisha Valerie Vanessa
Vera Veronica Vicky Victoria Violet Virginia Vivian Wanda Wendy Whitney Yvonne Zoe
"""

MALE = """
Aaron Abraham Adam Adrian Aidan Alan Albert Alec Alfred Allen Alvin Andre Andrew Andy Angelo
Anthony Antonio Archie Arnold Arthur Austin Barry Ben Benjamin Bernard Bill Billy Blake Bob Bobby
Brad Bradley Brandon Brendan Brent Brett Brian Bruce Bryan Caleb Calvin Carl Carlos Cesar Chad
Charles Charlie Chris Christopher Clarence Clark Claude Clayton Clifford Clint Clyde Cody
Colin Conor Corey Craig Curtis Dale Damian Dan Daniel Danny Darren Darryl Dave David Dean Dennis
Derek Derrick Desmond Devin Dominic Don Donald Doug Douglas Drew Duane Dustin Dwayne Dylan Earl
Eddie Edgar Edward Edwin Elias Elliot Eric Erik Ernest Ethan Eugene Evan Everett Felix Fernando
Floyd Francis Frank Franklin Fred Frederick Gabriel Garrett Gary Gavin Geoffrey George Gerald
Gilbert Glen Glenn Gordon Grant Greg Gregory Guy Hank Harold Harry Harvey Hector Henry Herbert
Howard Hugh Hunter Ian Isaac Ivan Jack Jackson Jacob Jake James Jamie Jared Jason Javier Jay Jeff
Jeffrey Jeremy Jerome Jerry Jesse Jim Jimmy Joe Joel John Johnny Jon Jonathan Jordan Jorge Jose
Joseph Josh Joshua Juan Julian Justin Karl Keith Kelvin Ken Kenneth Kent Kevin Kirk Kurt Kyle Lance
Larry Lawrence Lee Leo Leon Leonard Leroy Lester Levi Lewis Liam Lloyd Logan Louis Lucas Luke
Malcolm Manuel Marc Marcus Mario Mark Marshall Martin Marvin Mason Mathew Matt Matthew Maurice Max
Melvin Micah Michael Mike Miles Milton Mitchell Nathan Nathaniel Neil Nelson Nicholas Nick Noah
Norman Oliver Oscar Owen Patrick Paul Pedro Perry Pete Peter Philip Phillip Preston Quentin Ralph
Ramon Randall Randy Raul Ray Raymond Reggie Rex Ricardo Richard Rick Ricky Rob Robert Roberto
Rodney Roger Roland Ron Ronald Ronnie Ross Roy Ruben Rudy Russell Ryan Sam Samuel Saul Scott Sean
Seth Shane Shaun Sheldon Sidney Simon Spencer Stanley Stephen Steve Steven Stewart Stuart Ted
Terrance Terry Theodore Thomas Tim Timothy Tobias Todd Tom Tommy Tony Travis Trevor Troy Tyler
Vernon Victor Vincent Wade Wallace Walter Warren Wayne Wesley Wilbur Will William Willie Wyatt
Zachary
"""

UNISEX = """
Alex Ali Ash Avery Bailey Blair Cameron Casey Charlie Chris Dakota Dana Devon Drew Ellis Emerson
Finley Frankie Gray Harper Hayden Jaime Jamie Jesse Jo Jody Jordan Jules Kai Kelly Kendall Kerry
Lee Leslie Logan Mackenzie Max Morgan Nico Parker Pat Payton Quinn Reese Riley River Robin Rowan
Ryan Sage Sam Sawyer Shannon Shawn Sidney Skyler Sydney Taylor Terry Toni Tracy Val
"""


def build(block):
    return {w for w in block.split() if w}


F, M, U = build(FEMALE), build(MALE), build(UNISEX)
# A name appearing on both single-gender lists is treated as unisex.
overlap = F & M
U |= overlap
F -= U
M -= U
GENDER = {n: "f" for n in F}
GENDER.update({n: "m" for n in M})
GENDER.update({n: "u" for n in U})

humans = [json.loads(l) for l in open(f"{ROOT}/phase3/aita/data/aita_human.jsonl")]
seen = Counter()
for r in humans:
    for n in names_in(r["human_answer"]):
        if n in GENDER:
            seen[n] += 1

pool = [{"name": n, "gender": GENDER[n], "human_count": k}
        for n, k in sorted(seen.items(), key=lambda x: (-x[1], x[0]))]
out = f"{ROOT}/phase3/aita/data/name_pool.json"
json.dump({"source": "names on the curated gendered list that appear in the "
                     "human AITA corpus, weighted by their human frequency",
           "n_documents_scanned": len(humans),
           "pool": pool}, open(out, "w"), indent=1)

tot = sum(seen.values())
print(f"scanned {len(humans)} human documents")
print(f"curated list: {len(F)} female, {len(M)} male, {len(U)} unisex")
print(f"pool: {len(pool)} names, {tot} human mentions")
print(f"  by gender: " + ", ".join(
    f"{g} {sum(1 for p in pool if p['gender']==g)}" for g in "fmu"))
print(f"  top: " + ", ".join(f"{p['name']} {p['human_count']}" for p in pool[:10]))
print(f"  used once only: {sum(1 for p in pool if p['human_count']==1)}")
print(f"-> {out}")
