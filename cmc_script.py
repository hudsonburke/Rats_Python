import opensim as osim

setup_file = "/home/hudson/Rats_Python/Data/BAA01_Baseline_Walk05cmc_setup.xml"
cmcTool = osim.CMCTool(setup_file)
cmcTool.setResultsDir("/home/hudson/Rats_Python/Data")

cmcTool.run()