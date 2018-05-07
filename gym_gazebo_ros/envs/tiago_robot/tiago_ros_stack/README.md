# ros stack for assebmler

This stack will include all needed message and action types based ros middleware for commuincation, and also include the needed transform function, and exposed as pyhton interface. All used resources should be installed in the appropriate directory which can be easily.

Currently, we cannot find a better way to do these.

We can make this out of this package as a standalone package, because it can be enclosed. We install this ros stack on the system firstly(source the catkin_ws firstly in the same terminal), and then we can use python package which depend on this pacakge.

But how do we ensure the message defined from ROS pacakges can be used in non-ros python scripts? Something like `std_srvs.srv`.

And if we put some ros scripts here, we will easily call the ros module easiy not just the topics, and this will make the program easy. So how to put the auxiliary ros packages should be considered. And we should consider how to use ros pacakge as a general python package or c++ package which can be easily used out of ros. We know that gym use `proto` to make the communication out of ros.

Then the `examples` in `gym-gazebo-ros` will depend on the ros pakcages and the python module `gym_gazebo_ros`.