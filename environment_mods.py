def add_changes(model):
    # Add a static sphere
    model.worldbody.add(
        "geom",
        type="cylinder",
        size=[0.1, 0.1],  # Radius of the sphere
        pos=[0.5, 0.5, 0.1],  # Position [x, y, z]
        rgba=[1, 0, 1, 1],
    )  # Red color
    # model.worldbody.add()

    # # Add a movable sphere
    # sphere_body = model.worldbody.add('body', name='movable_sphere')
    # sphere_body.add('joint', type='free')
    # sphere_body.add('geom',
    #                 type='sphere',
    #                 size=[0.1],
    #                 pos=[1.5, 0.5, 0.1],
    #                 rgba=[0, 1, 0, 1],  # Green color
    #                 mass=1)
