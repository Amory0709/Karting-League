using KartGame.KartSystems;
using UnityEngine;

/// <summary>
/// This class inherits from TargetObject and represents a LapObject.
/// </summary>
public class LapObject : TargetObject
{
    [Header("LapObject")]
    [Tooltip("Is this the first/last lap object?")]
    public bool finishLap;

    [HideInInspector]
    public bool lapOverNextPass;

    void Start() {
        Register();
    }
    
    void OnEnable()
    {
        lapOverNextPass = false;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!((layerMask.value & 1 << other.gameObject.layer) > 0 && other.CompareTag("Player")))
            return;

        // Get the reference to the kart player script/component
        ArcadeKart kartPlayer = other.gameObject.GetComponentInParent<ArcadeKart>();

        if (kartPlayer != null)
        {
            Objective.OnUnregisterPickup?.Invoke(this, kartPlayer);
        }
        
    }
}
