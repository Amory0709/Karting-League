using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;
using System;
using Unity.MLAgents.Policies;

namespace KartGame.AI
{
    /// <summary>
    /// The KartAgent will drive the inputs for the KartController. This is a template, please implement your own agent by completing the TODO tasks,
    /// but don't be limited by this template, you could change anything in this script to get a better result.
    /// Tips: properties of KartAgent could also be set to public so that you can edit their values in Unity Editor. 
    /// !!! But remember to update your final settings to this script. In competition environment we only read the settings from this script.
    /// </summary>
    // TODO should change class name to KartAgent plus your team name abbreviation,e.g. KartAgentWoW   -DONE
    public class KartAgentSG : Agent, IInput
    {
        #region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        // use Training mode when you train model, use inferencing mode for competition
        // TODO should be set to Inferencing when you submit this script
        public AgentMode Mode = AgentMode.Training;

        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing. It is set to a random number in training mode")]
        public ushort InitCheckpointIndex = 0;
        #endregion

        #region Senses
        [Header("Observation Params")]
        [Tooltip("What objects should the raycasts hit and detect?")]
        public LayerMask Mask; // init in `InitializeSenses`

        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        public Sensor[] Sensors;

        [Header("Checkpoints")]
        [Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        public Collider[] Checkpoints;

        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        public LayerMask CheckpointMask; // init in `InitializeSenses`

        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? If not assigned, then the root transform will be used.")]
        public Transform AgentSensorTransform;
        #endregion

        #region Rewards
        // TODO Define your own reward/penalty items and set their values
        // For example:
        [Header("Rewards")]
        [Tooltip("Hit Penalty. What penalty is given when the agent crashes?")]
        public float HitPenalty = -1f;

        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        public float PassCheckpointReward = 1f;

        public float TowardsCheckpointRewards = 0.03f;

        public float SpeedReward = 0.02f;

        public float AccelerationReward = 0.0f;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        public LayerMask OutOfBoundsMask;

        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        public LayerMask TrackMask;

        [Tooltip("How far should the ray be when cast? For larger karts - this value should be larger too.")]
        public float GroundCastDistance;
        #endregion

        #region Debugging
        [Header("Debug Option")]
        [Tooltip("Should we visualize the rays that the agent draws?")]
        public bool ShowRayCast = true;
        #endregion


        private ArcadeKart m_Kart;
        private bool m_Acceleration;
        private bool m_Brake;
        private float m_Steering;
        private int m_CheckpointIndex;

        private bool m_EndEpisode;
        private float m_LastAccumulatedReward;

        private void Awake()
        {
            
            
            m_Kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null) AgentSensorTransform = transform;
            SetBehaviorParameters();
            SetDecisionRequester();
        }

        private void SetBehaviorParameters(){
            var behaviorParameters = GetComponent<BehaviorParameters>();
            behaviorParameters.UseChildSensors = true;
            behaviorParameters.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            // TODO set other behavior parameters according to your own agent implementation, especially following ones:
            behaviorParameters.BehaviorType = BehaviorType.Default;
            behaviorParameters.BehaviorName = "First-SG-Driver";
            behaviorParameters.BrainParameters.VectorObservationSize = 12; // size of the ML input data, should be the same as the `AddObservation` number
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 4;
            behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 3); // format of the ML model output data [0, 1, 2] [0, 1, 2]
            // continuous are floats [-1, 1]
        }

        private void SetDecisionRequester()
        {
            var decisionRequester = GetComponent<DecisionRequester>();
            //TODO set your decision requester
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.TakeActionsBetweenDecisions = true;
        }

        private void InitialiseResetParameters()
        {
            OutOfBoundsMask = LayerMask.GetMask("Ground");
            TrackMask = LayerMask.GetMask("Track");
            GroundCastDistance = 1f;
        }

        private void InitializeSenses()
        {
            // TODO Define your own sensors
            // make sure you are choosing from our predefined sensors, otherwise it won't work in competition
            // predefined:
            // "MLAgent_Sensors/0" "MLAgent_Sensors/15" "MLAgent_Sensors/30" "MLAgent_Sensors/45" "MLAgent_Sensors/60" "MLAgent_Sensors/75" "MLAgent_Sensors/90"
            // "MLAgent_Sensors/-15" "MLAgent_Sensors/-30" "MLAgent_Sensors/-45" "MLAgent_Sensors/-60" "MLAgent_Sensors/-75" "MLAgent_Sensors/-90"
            Sensors = new Sensor[9];
            Sensors[0] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-90"),
                HitValidationDistance = 4f,
                RayDistance = 5
            };
            Debug.Log(Sensors[0]);
            Sensors[1] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-60"),
                HitValidationDistance = 5f,
                RayDistance = 10
            };
            Sensors[2] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-30"),
                HitValidationDistance = 10f,
                RayDistance = 15
            };
            Sensors[3] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/0"),
                HitValidationDistance = 15f,
                RayDistance = 30
            };
            Sensors[4] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/30"),
                HitValidationDistance = 10f,
                RayDistance = 15
            };
            Sensors[5] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/60"),
                HitValidationDistance = 5f,
                RayDistance = 10,

            };
            Sensors[6] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/90"),
                HitValidationDistance = 4f,
                RayDistance = 5
            };
            Sensors[7] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/45"),
                HitValidationDistance = 7f,
                RayDistance = 12.5f
            };
            Sensors[8] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-45"),
                HitValidationDistance = 7f,
                RayDistance = 12.5f
            };

            // set Mask
            Mask = LayerMask.GetMask("Default", "Ground", "Environment", "Track");

            // set Checkpoints
            Checkpoints = GameObject.Find("Checkpoints").transform.GetComponentsInChildren<Collider>();

            // set CheckpointMask
            CheckpointMask = LayerMask.GetMask("CompetitionCheckpoints", "TrainingCheckpoints");
        }

        private void Start()
        {
            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent.
            OnEpisodeBegin();

            if (Mode == AgentMode.Inferencing)
                m_CheckpointIndex = InitCheckpointIndex;
        }

        private void Update()
        {
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                AddReward(m_LastAccumulatedReward);
                EndEpisode();
                OnEpisodeBegin();
            }
        }

        private void LateUpdate()
        {
            Debug.Log("Late Update detected.");
            switch (Mode)
            {
                case AgentMode.Inferencing:
                    if(ShowRayCast)
                    {
                        Debug.DrawRay(transform.position, Vector3.down * GroundCastDistance, Color.white);
                    }
                    // We want to place the agent back on the track if the agent happens to launch itself outside of the track.
                    if (Physics.Raycast(transform.position + Vector3.up, Vector3.down, out var hit, GroundCastDistance, TrackMask)
                        && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                    {
                        // Reset the agent back to its last known agent checkpoint
                        var checkpoint = Checkpoints[m_CheckpointIndex].transform;
                        transform.localRotation = checkpoint.rotation;
                        transform.position = checkpoint.position;
                        m_Kart.Rigidbody.velocity = default;
                        m_Steering = 0f;
                        m_Acceleration = m_Brake = false;
                    }
                    break;
            }
        }


        

        private void FindCheckPointIndex(Collider checkPoint, out int index)
        {
            for (int i = 0; i < Checkpoints.Length; i++)
            {
                if (Checkpoints[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }   {
            // TODO implement what should the agent do when it touched a checkpoint
            Debug.Log("!!!!!!!!!!!!!!!CONGRATULATION");
            Debug.Log("PASS ONE CHECKPOINT!!!");

            // ????
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;

            FindCheckPointIndex(other, out var index);


            if (triggered > 0 && index > m_CheckpointIndex || index == 0 && m_CheckpointIndex == Checkpoints.Length)
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;
            }
        }

        // input of the ML model
        // Called every step that the Agent requests a decision. This is one possible way for collecting the Agent's observations of the environment
        public override void CollectObservations(VectorSensor sensor)
        {
            // TODO Add your observations
            sensor.AddObservation(m_Kart.LocalSpeed());
            var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
            var nextCheckpoint = Checkpoints[next];
            if(nextCheckpoint != null) { return; }

            var direction = (nextCheckpoint.transform.position - m_Kart.transform.position).normalized;
            sensor.AddObservation(Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction));

            if(ShowRayCast)
            {
                Debug.DrawLine(AgentSensorTransform.position, nextCheckpoint.transform.position, Color.magenta);
            }

            m_LastAccumulatedReward = 0.0f;
            m_EndEpisode = false;

            foreach (var current in Sensors)
            {
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
               
                if(ShowRayCast)
                {
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.RayDistance, Color.green);
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.HitValidationDistance, Color.red);

                    if (hit && hitInfo.distance < current.HitValidationDistance) {
                        Debug.DrawRay(hitInfo.point, Vector3.up * 3.0f, Color.blue);
                    }
                }

                if(hit)
                {
                    if(hitInfo.distance < current.HitValidationDistance)
                    {
                        m_LastAccumulatedReward += HitPenalty;
                        m_EndEpisode = true;
                    }
                }

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
            }
            sensor.AddObservation(m_Acceleration);
        }

        // to control the movement of arcade kart
        public InputData GenerateInput()
        {
            return new InputData
            {
                Accelerate = m_Acceleration,
                Brake = m_Brake,
                TurnInput = m_Steering
            };
        }        

        // 1. Called at the beginning of an Agent's episode, including at the beginning of the simulation.
        public override void OnEpisodeBegin()
        {
            InitialiseResetParameters();
            InitializeSenses();
            switch (Mode)
            {
                case AgentMode.Training:
                    m_CheckpointIndex = Random.Range(0, Checkpoints.Length - 1);
                    var collider = Checkpoints[m_CheckpointIndex];
                    transform.localRotation = collider.transform.rotation;
                    transform.position = collider.transform.position;
                    m_Kart.Rigidbody.velocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    m_Steering = 0f;
                    break;

                default:
                    break;
            }
        }

        void OnTriggerEnter(Collider other)
        {
            // TODO implement what should the agent do when it touched a checkpoint
            Debug.Log("!!!!!!!!!!!!!!!CONGRATULATION");
            Debug.Log("PASS ONE CHECKPOINT!!!");

            // ????
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;

            FindCheckPointIndex(other, out var index);


            if (triggered > 0 && index > m_CheckpointIndex || index == 0 && m_CheckpointIndex == Checkpoints.Length)
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;
            }
        }
        
        private void FindCheckPointIndex(Collider checkPoint, out int index)
        {
            for (int i = 0; i < Checkpoints.Length; i++)
            {
                if (Checkpoints[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }

        // input of the ML model
        // Called every step that the Agent requests a decision. This is one possible way for collecting the Agent's observations of the environment
        public override void CollectObservations(VectorSensor sensor)
        {
            // TODO Add your observations
            sensor.AddObservation(m_Kart.LocalSpeed());
            var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
            var nextCheckpoint = Checkpoints[next];
            if(nextCheckpoint != null) { return; }

            var direction = (nextCheckpoint.transform.position - m_Kart.transform.position).normalized;
            sensor.AddObservation(Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction));

            if(ShowRayCast)
            {
                Debug.DrawLine(AgentSensorTransform.position, nextCheckpoint.transform.position, Color.magenta);
            }

            m_LastAccumulatedReward = 0.0f;
            m_EndEpisode = false;

            foreach (var current in Sensors)
            {
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
            
                if(ShowRayCast)
                {
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.RayDistance, Color.green);
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.HitValidationDistance, Color.red);

                    if (hit && hitInfo.distance < current.HitValidationDistance) {
                        Debug.DrawRay(hitInfo.point, Vector3.up * 3.0f, Color.blue);
                    }
                }

                if(hit)
                {
                    if(hitInfo.distance < current.HitValidationDistance)
                    {
                        m_LastAccumulatedReward += HitPenalty;
                        m_EndEpisode = true;
                    }
                }

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
            }
            sensor.AddObservation(m_Acceleration);
        }

        // the output of the AI model, the vector(??) of next movement
        // Called every time the Agent receives an action to take. Receives the action chosen by the Agent. It is also common to assign a reward in this method.
        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);
            // TODO Add your rewards/penalties

            var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
            var nextCheckPoint = Checkpoints[next];
            var direction = (nextCheckPoint.transform.position - m_Kart.transform.position).normalized;
            var reward = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction);

            if (ShowRayCast)
            {
                Debug.DrawRay(AgentSensorTransform.position, m_Kart.Rigidbody.velocity, Color.blue);
            }

            AddReward(reward * TowardsCheckpointRewards);
            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * AccelerationReward);
            AddReward(m_Kart.LocalSpeed() * SpeedReward);
        }

        private void InterpretDiscreteActions(ActionBuffers actions)
        {
            Debug.Log("Action 0: " + actions.DiscreteActions[0]); // action 1 means turn or not
            Debug.Log("Action 1: " + actions.DiscreteActions[1]); // action 2 means acceleration or not

            m_Steering = actions.DiscreteActions[0] - 1f;
            m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
            m_Brake = actions.DiscreteActions[1] < 1.0f;
        }

        /**
          When the Behavior Type is set to Heuristic Only in the Behavior Parameters of the Agent, 
          the Agent will use the Heuristic() method to generate the actions of the Agent. 
          As such, the Heuristic() method writes to the array of floats provided to the Heuristic 
          method as argument. Note: Do not create a new float array of action in the Heuristic() method, 
          as this will prevent writing floats to the original action array.
         */
        //public override void Heuristic(in ActionBuffers actionsOut)
        //{
        //    base.Heuristic(actionsOut);
        //    ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        //    discreteActions[0] = (int)Input.GetAxis("Vertical"); // x
        //    discreteActions[1] = (int)Input.GetAxis("Horizontal");// z
        //    // can be  created from user input
        //}

        
    }
}
