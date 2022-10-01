import os 
import json



def load_config(file_name):
    #checking if file exists or not
    if not os.path.exists(file_name):
        print("config file doesn't exists")
        print("creting config file..........")
        config_template = {
                            "avg_heights":{
                                "car":None,
                                "bike":None,
                                "bottle":None
                                },
                            "object_measured_distance":None,
                            "focal_length":None,
                            "thresholds":[{
                                "emergency_braking":{
                                    "distance":None,
                                    "speed":None
                                },
                                "alert":{
                                    "distance":None,
                                    "speed": None
                                }
                                }],
                                "confidence":None
                            }

        #creting config file
        with open(file_name,"w") as f:
            json.dump(config_template, f)

    #loading json file
    template_json = None
    with open(file_name,"r") as f:
        template_json = json.load(f)
    return template_json


# if __name__ =="__main__":
#     load_config("config.json")