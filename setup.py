from setuptools import find_packages,setup

HYPEN='-e .'
def read_requirements(file:str):
    """
    This functions takes the file as an input and read it store the line in the list 
    """
    requirement=[]
    with open(file) as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace('\n','') for req in requirement]
        if HYPEN in requirement:
            requirement.remove(HYPEN)


                
    return requirement    
    




setup(
    name="PROJ1",
    version="0.0.1",
    author="Krish",
    author_email="krishsharma5272@gamil.com",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt')
)


