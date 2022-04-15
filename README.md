# GV-helpers

Helpers modules for isatis.neo.

## Instructions

Add the code below to your calculator initialization script.

    if app[ 'project_batch_path' ] not in sys.path: 
        sys.path.append(app[ 'project_batch_path'])

Then, import the desired module as shown below. The file must be in the same folder as the batch file.

    import DDH_BH_analysis